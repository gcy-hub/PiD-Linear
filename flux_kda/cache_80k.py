"""Three-GPU, resumable native FLUX3 cache for the 80K 1K aspect buckets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPUS=(1,2,3)
os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,GPUS))
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'flux3/DiffSynth-Studio'))
DATA=Path(os.environ.get('FLUX3_DATASET_ROOT', 'dataset/MultiAspect-4K-1M')).resolve()
ROOT=DATA/'flux3_qdm_cache_80k_1k'
BUCKETS={'1,1':(1024,1024),'4,3':(1152,864),'3,4':(864,1152),'16,9':(1344,768),'9,16':(768,1344),'3,2':(1248,832),'2,3':(832,1248)}


def setup():
    from flux3_qdm import data as d
    from flux3_qdm.prepare import component_fingerprints
    from flux3_qdm.resources import resource_manifest
    d.IMAGE_PREPROCESS['buckets']=BUCKETS
    resources=resource_manifest(Path(os.environ.get('FLUX3_WEIGHTS_ROOT', 'weights')))
    store=d.CacheStore(ROOT,component_fingerprints(resources))
    rows=d.load_records(DATA/'pid_vdn_subsets',80000)
    return d,store,rows,resources


def worker(rank,phase,smoke=False):
    import torch
    import numpy as np
    from flux3_qdm.prepare import _load_phase
    from flux3_qdm.data import prompt_ids
    torch.set_num_threads(4);torch.cuda.set_device(rank)
    device=f'cuda:{rank}'
    d,store,rows,_=setup()
    pipe=_load_phase(phase,Path(os.environ.get('FLUX3_WEIGHTS_ROOT', 'weights')),device)
    done=0;tested=False
    def save_text(keys,inputs,length):
        nonlocal done,tested
        mask=inputs['attention_mask'].to(device)
        positions=(mask.long().cumsum(-1)-1).masked_fill(mask==0,1)
        hidden=pipe.text_encoder(input_ids=inputs['input_ids'].to(device),attention_mask=mask,
            position_ids=positions[None].expand(4,-1,-1),use_cache=False)
        tokens=torch.stack([hidden[i] for i in (4,8,12,16,20,24,28,32)],dim=2).reshape(len(keys),length,20480).to(torch.bfloat16)
        if not tested:
            from diffsynth.pipelines.flux3_image import Flux3Unit_PromptEmbedder
            reference=Flux3Unit_PromptEmbedder().process(pipe,keys[0][1])['prompt_tokens']
            error=(tokens[0].float()-reference.float()).square().mean()/reference.float().square().mean()
            assert error.item()<0.0004, error.item()
            tested=True
        for (key,_),token in zip(keys,tokens):
            assert torch.isfinite(token).all()
            store.write('text',key,key,dict(prompt_tokens=token,prompt_ids=prompt_ids(length)))
        done+=len(keys)
    with torch.inference_mode():
        if phase=='image':
            pending=[r for r in rows[rank::len(GPUS)] if not store.valid('image',store.keys(r)[0],d.record_fingerprint(r))]
            groups={b:[] for b in BUCKETS}
            for row in pending:groups[row['bucket']].append(row)
            def prepare(row):return np.array(d.prepare_image(row['image_path'],BUCKETS[row['bucket']]),dtype=np.uint8,copy=True)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for bucket,group in groups.items():
                    if smoke:group=group[:20]
                    for start in range(0,len(group),20):
                        batch=group[start:start+20]
                        images=torch.from_numpy(np.stack(list(pool.map(prepare,batch)))).to(device=device,dtype=torch.bfloat16)
                        images=(images*(2/255)-1).permute(0,3,1,2)
                        latent=pipe.vae.encode(images)
                        w,h=BUCKETS[bucket]
                        assert latent.shape==(len(batch),128,h//16,w//16) and torch.isfinite(latent).all()
                        if not tested:
                            reference=pipe.vae.encode(images[:1]);error=(latent[:1].float()-reference.float()).square().mean()/reference.float().square().mean()
                            assert error.item()<0.0004,error.item()
                            tested=True
                        for row,z in zip(batch,latent):
                            store.write('image',store.keys(row)[0],d.record_fingerprint(row),dict(image_latents=z.permute(1,2,0).reshape(-1,128),image_ids=d.image_ids(h//16,w//16),grid=(h//16,w//16)))
                        done+=len(batch)
                        if done<=len(batch) or done%100<len(batch):print(json.dumps(dict(phase=phase,gpu=GPUS[rank],done=done,total=len(pending),batch=len(batch),time=time.time(),peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        else:
            captions=list(dict.fromkeys(['']+[r['caption'] for r in rows]))[rank::len(GPUS)]
            pending=[(d.fingerprint(c),c) for c in captions if not store.valid('text',d.fingerprint(c),d.fingerprint(c))]
            if smoke:pending=pending[:128]
            groups={}
            tokenizer=pipe.processor.tokenizer
            for key,caption in pending:
                text=pipe.processor.apply_chat_template([{'role':'user','content':caption}],tokenize=False,add_generation_prompt=True)
                real=len(tokenizer(text,truncation=True,max_length=4096)['input_ids'])
                length=min(((real+79)//80)*80,4096)
                groups.setdefault(length,[]).append((key,caption,text))
            for length,group in sorted(groups.items()):
                batch_size=max(1,min(32,5120//length))
                for start in range(0,len(group),batch_size):
                    batch=group[start:start+batch_size]
                    inputs=tokenizer([x[2] for x in batch],return_tensors='pt',padding='max_length',truncation=True,max_length=length)
                    save_text([(x[0],x[1]) for x in batch],inputs,length)
                    if done%100<batch_size:print(json.dumps(dict(phase=phase,gpu=GPUS[rank],done=done,total=len(pending),batch=batch_size,length=length,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
    print(json.dumps(dict(phase=phase,gpu=GPUS[rank],done=done,complete=True,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)


def wait_workers(children):
    while True:
        codes=[p.poll() for p in children]
        if any(code not in (None,0) for code in codes):
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:p.wait()
            return codes
        if all(code is not None for code in codes):return codes
        time.sleep(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--phase',choices=['image','text'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.worker is not None:worker(args.worker,args.phase,args.smoke);return
    import socket
    host = os.environ.get('FLUX_KDA_HOST')
    if host and socket.gethostname().split('.')[0] != host:return
    if any((ROOT/x).exists() for x in ['COMPLETE','STOP','FAILED']):return
    with (ROOT/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if any(int(free.splitlines()[gpu])<40960 for gpu in GPUS):return
        if not (ROOT/'text').exists():(ROOT/'text').symlink_to(DATA/'flux3_qdm_cache/text',target_is_directory=True)
        d,store,rows,resources=setup()
        store.initialize_subset(DATA/'pid_vdn_subsets',80000,rows,resources=dict(weights_root=str((Path(os.environ.get('FLUX3_WEIGHTS_ROOT', 'weights'))).resolve()),fingerprints=resources['fingerprints'],revision=resources['revision']))
        for phase in ['image','text']:
            children=[]
            for rank,gpu in enumerate(GPUS):
                log=(ROOT/f'{phase}-gpu{gpu}.log').open('a')
                child=subprocess.Popen([sys.executable,'-u','-m','flux_kda.cache_80k','--worker',str(rank),'--phase',phase]+(['--smoke'] if args.smoke else []),cwd=REPO,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                log.close();children.append(child)
            codes=wait_workers(children)
            if any(code not in (None,0) for code in codes):
                if any(c not in (None,0,-15,-9,137,143) for c in codes):(ROOT/'FAILED').write_text(str(codes))
                raise SystemExit(1)
        if args.smoke:
            (ROOT/'SMOKE_COMPLETE').touch()
            return
        d.CachedDataset(DATA/'pid_vdn_subsets',ROOT,80000)
        (ROOT/'COMPLETE').write_text(json.dumps(dict(rows=80000,buckets=BUCKETS,gpus=GPUS,image_batch=20,text_max_batch=32,time=time.time()))+'\n')


if __name__=='__main__':main()
