"""Multi-GPU native PixelDiT generation from existing Gemma caption caches."""
import argparse
import fcntl
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.evaluation import tensor_image
from pid._src.linear_pid.text_cache import TextCacheReader
from pid._src.linear_pid.training import init_distributed
from .conditioning import load_null_cache
from .runtime import load_net, sample
from .synthetic_data import generated_complete, publish_dataset, read_records, save_generated


def generate(args):
    started=time.monotonic()
    torch.set_num_threads(args.threads)
    index=Path(args.index_root).resolve();root=index.parent;run=Path(args.run_root).resolve()
    stage=run/'a1';stage.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'generation.json').read_text());settings=meta['settings']
    records=read_records(index)
    if args.limit:records=records[:args.limit]
    rank,world,device=init_distributed()
    lock=None
    try:
        failure=None
        if rank==0:
            try:
                lock=(root/'generation.lock').open('a')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if sha256_file(args.teacher_checkpoint)!=settings['base_sha256']:
                    raise ValueError('Synthetic generator base weight fingerprint changed')
            except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
        if world>1:
            values=[failure];dist.broadcast_object_list(values,src=0);failure=values[0]
        if failure:raise RuntimeError(failure)
        if all(generated_complete(root,r,meta['fingerprint']) for r in records):
            if rank==0:
                completed=len(list((root/'images').glob('*.done.json')))
                complete=completed==meta['train_size'] and publish_dataset(root,args.workers)
                atomic_json(stage/'synthetic-progress.json',dict(completed=completed,total=meta['train_size'],
                            phase='ready' if complete else 'generation',elapsed_seconds=time.monotonic()-started))
                print(f'All {len(records)} requested images already complete; DiT was not loaded',flush=True)
            if world>1:dist.barrier()
            return
        cache=TextCacheReader(args.text_cache_root)
        if cache.fingerprint!=settings['text_cache_fingerprint']:
            raise ValueError('Synthetic caption cache fingerprint changed')
        special=load_null_cache(args.text_cache_root)
        null_embs=special['null_embs'].to(device);null_mask=special['null_mask'].to(device)
        net=load_net(args.teacher_checkpoint).to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
        # Deliberately no QDM conversion, no Gemma model, no VAE and no offload.
        batch_size=args.batch_size or settings['generation_batch_size']
        if rank==0:print(f'Generation: {world} GPUs, batch {batch_size} per GPU',flush=True)
        tasks=[records[i:i+batch_size] for i in range(0,len(records),batch_size)]
        emitted=0
        last_batch_seconds=0
        claims=root/'batch-locks';claims.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool, torch.no_grad():
            # All GPUs consume one queue. A faster GPU can claim additional batches;
            # flock releases automatically on exit/kill, so no stale claims survive.
            for number,group in tqdm(enumerate(tasks),total=len(tasks),desc=f'Original PixelDiT rank{rank}',unit='batch'):
                if (stage/'STOP').exists() or (args.max_seconds and time.monotonic()-started>=args.max_seconds):break
                if args.max_seconds and last_batch_seconds and (
                    time.monotonic()-started+last_batch_seconds*1.15+10>=args.max_seconds
                ):break
                missing=[not generated_complete(root,r,meta['fingerprint']) for r in group]
                if not any(missing):continue
                with (claims/f'{number:06d}.lock').open('a') as claim:
                    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    missing=[not generated_complete(root,r,meta['fingerprint']) for r in group]
                    if not any(missing):continue
                    group=[record for record,needed in zip(group,missing) if needed]
                    embeddings,masks=zip(*(cache.read(r['id'],r['caption']) for r in group))
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device);tick=time.perf_counter()
                    images=sample(net,torch.stack(embeddings).to(device),torch.stack(masks).to(device),
                                  null_embs,null_mask,height=settings['height'],width=settings['width'],
                                  seed=[r['seed'] for r in group],steps=settings['sampling_steps'],
                                  cfg=settings['cfg'],shift=settings['shift'])
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite synthetic pixels')
                    with (stage/f'synthetic-batches-rank-{rank}.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(batch=number,images=len(group),batch_size=batch_size,seconds=seconds,
                                     peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30))+'\n')
                    futures=[]
                    for i,record in enumerate(group):
                        image=tensor_image(images[i:i+1])
                        futures.append(pool.submit(save_generated,root,record,image,meta['fingerprint']))
                        emitted+=1
                    for future in futures:future.result()
                    del images,embeddings,masks
                    last_batch_seconds=time.perf_counter()-tick
                if rank==0:
                    completed=len(list((root/'images').glob('*.done.json')))
                    atomic_json(stage/'synthetic-progress.json',dict(completed=completed,total=meta['train_size'],
                                elapsed_seconds=time.monotonic()-started,phase='generation'))
        if world>1:dist.barrier()
        if rank==0:
            completed=len(list((root/'images').glob('*.done.json')))
            complete=completed==meta['train_size'] and publish_dataset(root,args.workers)
            atomic_json(stage/'synthetic-progress.json',dict(completed=completed,total=meta['train_size'],
                        elapsed_seconds=time.monotonic()-started,phase='ready' if complete else 'generation'))
            print(json.dumps(dict(generated=completed,total=meta['train_size'],complete=complete,
                                  elapsed_seconds=time.monotonic()-started)),flush=True)
        atomic_json(stage/f'synthetic-rank-{rank}.json',dict(new_images=emitted,seconds=time.monotonic()-started))
    finally:
        if lock:lock.close()
        if dist.is_initialized():dist.destroy_process_group()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-root',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--teacher-checkpoint',required=True);p.add_argument('--text-cache-root',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--max-seconds',type=int,default=1200);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=0,help='Execution batch; zero uses the original preparation default')
    args=p.parse_args(argv)
    if min(args.workers,args.threads)<1 or min(args.max_seconds,args.limit,args.batch_size)<0:p.error('Invalid worker/time/limit/batch')
    generate(args)


if __name__=='__main__':main()
