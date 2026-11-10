"""Resume the paired 600-to-700-step mixing experiment through node cleanup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/flux3-mixing-ab'
SOURCE=REPO/'outputs/flux3-stock-kda-clear-node01-bs1'
PYTHON=sys.executable
ENV=dict(os.environ,PYTHONPATH=f'{REPO}:{REPO}/flux2:{REPO}/flux3',OMP_NUM_THREADS='4',
         FLA_DISABLE_TENSOR_CACHE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
DATA=os.environ.get('FLUX3_SUBSET_ROOT', 'dataset/CLEAR-FLUX3-klein-base-2K-2048')
WEIGHTS=os.environ.get('FLUX3_WEIGHTS', 'weights/flux-3-image-klein-base/model.safetensors')


def step(path):
    import torch
    return torch.load(path,map_location='cpu',mmap=True,weights_only=False)['step']


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (SOURCE/'STOP').exists():
        (ROOT/'resume-original').touch()
        (SOURCE/'STOP').touch()
    start=ROOT/'start';start.mkdir(exist_ok=True)
    if not (start/'latest.pt').exists():
        snapshot=REPO/'outputs/flux3-step600-1k-samples/checkpoint.pt'
        assert step(snapshot)==600
        os.link(snapshot,start/'latest.pt')
    for rank in range(3):
        name=f'optimizer-600-rank-{rank}.pt'
        if not (start/name).exists():os.link(SOURCE/name,start/name)
    for name in ['baseline','calibrated']:
        arm=ROOT/name;arm.mkdir(exist_ok=True)
        if not (arm/'latest.pt').exists():
            for path in start.glob('*.pt'):os.link(path,arm/path.name)
    (ROOT/'plan.json').write_text(json.dumps(dict(start_step=600,end_step=700,gpus=[1,2,3],
        batch_per_gpu=1,accumulation=1,seed=42,optimizer='restored FP32-master Prodigy',
        calibration='diagnostics/calibration.json',arms=['baseline','calibrated']),indent=2)+'\n')


def run(lock_fd):
    from PIL import Image,ImageDraw
    for name in ['baseline','calibrated']:
        arm=ROOT/name
        if step(arm/'latest.pt')<700:
            command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',
                '-m','flux_kda.train','--data-mode','cached','--shard-optimizer','--gradient-checkpointing','--family','flux3','--weights',WEIGHTS,
                '--subset-root',DATA,'--cache-root',DATA+'/cache-flux3-1024','--subset-size','2000',
                '--output',str(arm),'--resume',str(arm/'latest.pt'),'--steps','700',
                '--batch-size','1','--grad-accum','1','--save-every','25']
            if name=='calibrated':command+=['--text-logit-bias',str(ROOT/'diagnostics/calibration.json')]
            with (arm/'run.log').open('a') as log:
                subprocess.run(command,cwd=REPO,env=dict(ENV,CUDA_VISIBLE_DEVICES='1,2,3'),
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock_fd,),check=True)
        manifest=json.loads((REPO/'outputs/flux3-step600-1k-samples/manifest.json').read_text())
        def generate(index):
            directory=arm/f'images/{index+1:02d}';directory.mkdir(parents=True,exist_ok=True)
            if (directory/'kda.json').exists():return
            shutil.copyfile(REPO/f'outputs/flux3-step600-1k-samples/{index+1:02d}/conditions.pt',directory/'conditions.pt')
            command=[PYTHON,'-m','flux_kda.benchmark','--family','flux3','--weights',WEIGHTS,
                     '--checkpoint',str(arm/'latest.pt'),'--variant','kda','--size','1024',
                     '--steps','20','--cfg','3.5','--seed','0','--prompt',manifest['images'][index]['prompt'],
                     '--output',str(directory)]
            with (directory/'run.log').open('a') as log:
                subprocess.run(command,cwd=REPO,env=dict(ENV,CUDA_VISIBLE_DEVICES=str(index+1)),
                               stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock_fd,),check=True)
            with Image.open(directory/'kda.png') as image:assert image.size==(1024,1024)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(generate,range(3)))
    montage=Image.new('RGB',(1536,1104),'white');draw=ImageDraw.Draw(montage)
    for row,name in enumerate(['baseline','calibrated']):
        draw.text((12,row*552+12),f'{name} | step 700 | 100 additional steps',fill='black')
        for col in range(3):
            with Image.open(ROOT/name/f'images/{col+1:02d}/kda.png') as image:
                montage.paste(image.convert('RGB').resize((512,512)),(col*512,row*552+40))
    montage.save(ROOT/'comparison.jpg',quality=94)
    (ROOT/'DONE').write_text('Both arms completed 100 updates and three matched 1K samples.\n')
    if (ROOT/'resume-original').exists():(SOURCE/'STOP').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--run-fd',type=int)
    args=parser.parse_args()
    if args.prepare:prepare();return
    host = os.environ.get('FLUX_KDA_HOST')
    if host and socket.gethostname().split('.')[0] != host:return
    if args.run_fd is not None:
        try:run(args.run_fd)
        except Exception as exc:
            (ROOT/'FAILED.json').write_text(json.dumps(dict(error=repr(exc),time=time.time()))+'\n')
            raise
        finally:os.close(args.run_fd)
        return
    if not args.check:parser.error('Select --prepare or --check')
    if any((ROOT/name).exists() for name in ['DONE','FAILED.json','STOP']):return
    if not (ROOT/'heldout-check/COMPLETE').exists():return
    with (ROOT/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,check=True,timeout=15)
        free={int(line.split(',')[0]):int(line.split(',')[1]) for line in result.stdout.splitlines()}
        if any(free.get(i,0)<40960 for i in [1,2,3]):
            (ROOT/'status.json').write_text(json.dumps(dict(status='waiting-for-gpus',free_mib=free,time=time.time()))+'\n')
            return
        with (ROOT/'pipeline.log').open('a') as log:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--run-fd',str(lock.fileno())],
                cwd=REPO,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(lock.fileno(),))
        (ROOT/'status.json').write_text(json.dumps(dict(status='running',pid=child.pid,time=time.time()))+'\n')


if __name__=='__main__':main()
