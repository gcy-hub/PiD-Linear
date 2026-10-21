"""Renewable synthetic image preparation followed by PixelDiT A1 only."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pid._src.linear_pid.data import sha256_file
from pid._src.linear_pid.text_cache import TextCacheReader
from pid._src.vdn_pid.training import parse_args
from scripts.repeat_pid_vdn import launch, read_json
from .synthetic_data import prepare_dataset

REPO=Path(__file__).resolve().parents[3]


def cycle(argv):
    config,_=parse_args(argv)
    started=time.monotonic()
    index=Path(config.train_index_root).resolve();run=Path(config.output_root).resolve()
    if config.backbone!='pixeldit_t2i' or config.stage!='a1':
        raise ValueError('Synthetic pipeline runs native PixelDiT A1 only')
    ready=index/'index.json'
    if not ready.exists():
        visible=os.getenv('GPU_IDS',os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3')).split(',')
        command=[sys.executable,'-m','torch.distributed.run','--standalone',
                 '--nproc_per_node',str(len(visible)),'-m','pid._src.pixeldit_qdm.synthetic_generate',
                 '--index-root',str(index),'--run-root',str(run),'--teacher-checkpoint',config.teacher_path,
                 '--text-cache-root',config.text_cache_root,'--threads',str(config.threads),
                 '--workers',str(max(1,config.workers)),'--max-seconds',str(config.max_seconds)]
        print('PHASE original-model synthetic generation',flush=True)
        result=subprocess.call(command,cwd=REPO)
        if result:return result
    if not ready.exists() or (run/'a1/STOP').exists():return 0
    remaining=max(0,int(config.max_seconds-(time.monotonic()-started))) if config.max_seconds else 0
    if config.max_seconds and remaining<120:return 0
    print('PHASE synthetic-data A1 from original PixelDiT weights',flush=True)
    return subprocess.call(['bash',str(REPO/'scripts/train_pixeldit_qdm.sh'),*argv,
                            '--stage','a1','--max-seconds',str(remaining)],cwd=REPO)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='cycle':raise SystemExit(cycle(argv[1:]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','prepare','status','stop'],nargs='?',default='start')
    p.add_argument('--data-root',default='dataset/MultiAspect-4K-1M/pixeldit_t2i_synthetic_a1')
    p.add_argument('--generation-batch-size',type=int,default=4)
    p.add_argument('--cycle-seconds',type=int,default=1200)
    args,remaining=p.parse_known_args(argv)
    defaults=['--backbone','pixeldit_t2i','--memory-mode','fixed4','--local-mode','halo','--local-halo','2',
              '--region-size','8','--remote-rope','--resolution','1024','--flow-shift','4',
              '--shift-base-size','1024','--grad-accum','1','--stage','a1',
              '--output-root','outputs/pixeldit-t2i-qdm-halo2-rope-synthetic-a1']
    config,_=parse_args(defaults+remaining)
    root=Path(config.output_root).resolve()
    if args.action=='status':
        print(json.dumps(dict(controller=read_json(root/'repeat/state.json'),
                              generation=read_json(root/'a1/synthetic-progress.json')),indent=2));return
    if args.action=='stop':
        (root/'repeat').mkdir(parents=True,exist_ok=True);(root/'repeat/STOP').touch()
        print('Stop requested; generation/training saves at a batch boundary and will not restart');return
    if config.backbone!='pixeldit_t2i' or config.stage!='a1' or config.resolution!='1024':
        p.error('This pipeline supports PixelDiT A1 at native 1024 only')
    if not config.text_cache_root:p.error('Existing Gemma cache is required')
    if config.init_from:p.error('This experiment starts from original PixelDiT; omit --init-from')
    if args.generation_batch_size<1 or args.cycle_seconds<1:p.error('Positive generation batch/cycle required')
    world=len(os.getenv('GPU_IDS',os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3')).split(','))
    stage=config.stage_settings('a1',world)
    cache=TextCacheReader(config.text_cache_root)
    settings=dict(steps=stage['steps'],effective_batch=stage['effective'],seed=config.seed,height=1024,width=1024,
                  sampling_steps=50,cfg=2.75,shift=4.,generation_batch_size=args.generation_batch_size,
                  base_sha256=sha256_file(config.teacher_path),text_cache_fingerprint=cache.fingerprint)
    meta=prepare_dataset(Path(config.subset_root)/'20000',args.data_root,settings)
    config.train_index_root=str((Path(args.data_root)/'index').resolve())
    print(json.dumps(dict(exposures=meta['exposures'],images=meta['train_size'],
                          index=config.train_index_root,stage=stage)),flush=True)
    if args.action=='prepare':return
    training_args=[]
    for key,value in config.to_dict().items():
        if key in ('stage','output_root','max_seconds'):continue
        flag='--'+key.replace('_','-')
        if key=='activation_checkpointing':
            if not value:training_args.append('--no-activation-checkpointing')
        elif key=='remote_rope':training_args.append('--remote-rope' if value else '--no-remote-rope')
        elif key in ('allow_batch_change','allow_schedule_change'):
            if value:training_args.append(flag)
        else:training_args.extend((flag,str(value)))
    request=dict(stage='a1',cycle_seconds=args.cycle_seconds,grace_seconds=180,restart_delay=5,max_cycles=0,
                 driver=str(REPO/'scripts/train_pixeldit_synthetic_cycle.sh'),training_args=training_args)
    launch(root,request)


if __name__=='__main__':main()
