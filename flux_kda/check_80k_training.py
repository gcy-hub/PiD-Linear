"""Test all seven cached 1K buckets with the production trainer, then resume caching."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from flux3_qdm import data as d
from flux3_qdm.runtime import GlobalSampler

REPO=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('FLUX3_DATASET_ROOT', 'dataset/MultiAspect-4K-1M')).resolve()
CACHE=DATA/'flux3_qdm_cache_80k_1k'
ROOT=REPO/'../experiment/flux3-80k-training-smoke'
SOURCE=REPO/'../experiment/flux3-stock-kda-clear-node01-bs1/latest.pt'


def main():
    import torch
    ROOT.mkdir(exist_ok=False)
    subset=ROOT/'subset';subset.mkdir()
    testcache=ROOT/'cache';testcache.mkdir()
    manifest=json.loads((CACHE/'subsets/80000.json').read_text())
    store=d.CacheStore(CACHE,manifest['components'])
    records=d.load_records(DATA/'pid_vdn_subsets',80000)
    selected=[]
    for bucket in d.BUCKETS_1K:
        candidates=[]
        for row in records:
            if row['bucket']!=bucket:continue
            ik,tk=store.keys(row);ip=store.path('image',ik);tp=store.path('text',tk)
            if ip.with_suffix('.json').exists() and tp.with_suffix('.json').exists():
                candidates.append((tp.stat().st_size,row))
        for _,row in sorted(candidates,key=lambda x:x[0],reverse=True):
            ik,tk=store.keys(row)
            if store.valid('image',ik,d.record_fingerprint(row)) and store.valid('text',tk,tk):
                selected.append(row)
                if sum(r['bucket']==bucket for r in selected)==3:break
        assert sum(r['bucket']==bucket for r in selected)==3,bucket
    offsets=[]
    with (subset/'manifest.jsonl').open('wb') as stream:
        for row in selected:
            offsets.append(stream.tell());stream.write((json.dumps(row)+'\n').encode())
    np.save(subset/'offsets.npy',np.array(offsets,dtype=np.int64))
    for bucket in d.BUCKETS_1K:
        np.save(subset/f'train_{bucket.replace(",","_")}.npy',np.array([i for i,r in enumerate(selected) if r['bucket']==bucket],dtype=np.int64))
    (subset/'index.json').write_text(json.dumps(dict(train_size=len(selected),dataset_root=str(DATA))))
    ordered=d.load_records(subset,len(selected))
    for kind in ['image','text']:(testcache/kind).symlink_to(CACHE/kind,target_is_directory=True)
    original=d.IMAGE_PREPROCESS['buckets'];d.IMAGE_PREPROCESS['buckets']=d.BUCKETS_1K
    try:d.CacheStore(testcache,manifest['components']).initialize_subset(subset,len(selected),ordered,resources=manifest['resources'])
    finally:d.IMAGE_PREPROCESS['buckets']=original
    dataset=d.CachedDataset(subset,testcache,len(selected),image_buckets=d.BUCKETS_1K)
    for i in range(len(dataset)):
        sample=dataset[i]
        assert sample['grid']==tuple(reversed(tuple(x//16 for x in d.BUCKETS_1K[ordered[i]['bucket']])))
    try:d.CachedDataset(subset,testcache,len(selected))
    except RuntimeError as exc:assert 'preprocessing fingerprint' in str(exc)
    else:raise AssertionError('2K configuration accepted the 1K cache')
    state=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    start=state['step'];assert start==4000
    del state
    sampler=GlobalSampler(dataset.bucket_indices,1,3,1,42)
    covered=set();schedule=[]
    for step in range(start,start+100):
        ids=sampler.global_batch(step);bucket=ordered[ids[0]]['bucket'];covered.add(bucket)
        schedule.append(dict(step=step+1,bucket=bucket,ids=[ordered[i]['id'] for i in ids],text_lengths=[len(dataset[i]['prompt_tokens']) for i in ids]))
        if len(covered)==7:break
    assert len(covered)==7
    end=schedule[-1]['step']
    (ROOT/'schedule.json').write_text(json.dumps(schedule,indent=2))
    print(json.dumps(dict(start=start,end=end,schedule=schedule)),flush=True)
    output=ROOT/'training'
    common=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=3','-m','flux_kda.train',
            '--family','flux3','--data-mode','cached','--shard-optimizer','--gradient-checkpointing','--weights',os.environ.get('FLUX3_WEIGHTS', 'weights/flux-3-image-klein-base/model.safetensors'),
            '--subset-root',str(subset),'--cache-root',str(testcache),'--subset-size','21','--image-resolution','1024',
            '--output',str(output),'--batch-size','1','--grad-accum','1','--save-every','100000']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1,2,3',PYTHONPATH='.:flux2:flux3',OMP_NUM_THREADS='4',
             FLA_DISABLE_TENSOR_CACHE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    print('CPU cache checks passed; waiting for 40 GiB free on GPUs 1,2,3',flush=True)
    while True:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        if all(int(free[gpu])>=40960 for gpu in (1,2,3)):break
        time.sleep(30)
    for target,resume,extra in [(end,SOURCE,['--allow-dataset-change']),(end+1,output/'latest.pt',[])]:
        with (ROOT/f'train-to-{target}.log').open('w') as log:
            subprocess.run(common+['--steps',str(target),'--resume',str(resume)]+extra,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=torch.load(output/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
        assert checkpoint['step']==target and checkpoint['optimizer_shards']==3
        del checkpoint
        for rank in range(3):assert (output/f'optimizer-{target}-rank-{rank}.pt').is_file()
    metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==end+1-start and all(np.isfinite(r['loss']) for r in metrics)
    print(json.dumps(dict(status='passed',tested_buckets=sorted(covered),training_steps=len(metrics),
                         max_peak_gib=max(r['peak_allocated_gib'] for r in metrics),last_loss=metrics[-1]['loss'],restored_step=end)),flush=True)
    shutil.rmtree(ROOT)
    assert not ROOT.exists()
    (CACHE/'STOP').unlink()
    with (CACHE/'launcher.log').open('a') as log:
        subprocess.Popen([sys.executable,'-u','-m','flux_kda.cache_80k'],cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Test directory deleted; cache recovery released and cache launcher requested',flush=True)


if __name__=='__main__':main()
