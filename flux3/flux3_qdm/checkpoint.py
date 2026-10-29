"""Completed delta checkpoints; optimizer/RNG are training state, never CPU offload."""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'].cpu())
    if value.get('cuda') is not None:torch.cuda.set_rng_state(value['cuda'].cpu())


def is_complete(path):
    path=Path(path)
    try:
        meta=json.loads((path/'complete.json').read_text())
        return all((path/k).stat().st_size==n for k,n in meta['files'].items()) and {'state.pt','delta.pt'} <= set(meta['files'])
    except (OSError,ValueError,KeyError,TypeError):return False


def latest_checkpoint(stage_dir):
    paths=[p for p in (Path(stage_dir)/'checkpoints').glob('step_*') if not p.name.endswith('.incomplete') and is_complete(p)]
    return max(paths,key=lambda p:(int(json.loads((p/'complete.json').read_text())['step']),p.name)) if paths else None


def read_checkpoint(path):
    path=Path(path)
    if not is_complete(path):raise ValueError(f'Incomplete or corrupt checkpoint: {path}')
    return torch.load(path/'state.pt',map_location='cpu',weights_only=False,mmap=True)


def save_checkpoint(stage_dir,delta,optimizer,scheduler,progress,config,metadata,keep=3,rng=None,milestone=False):
    stage_dir=Path(stage_dir);root=stage_dir/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    step=int(progress['stage_step']);name=f'step_{step:09d}_{time.time_ns()}'
    pending=root/(name+'.incomplete');pending.mkdir()
    torch.save(cpu_tree(delta),pending/'delta.pt')
    payload=dict(schema=1,progress=progress,config=config,metadata=metadata,
                 optimizer=cpu_tree(optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer),
                 scheduler=scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler,
                 rng=rng if rng is not None else [capture_rng()])
    torch.save(payload,pending/'state.pt')
    for p in pending.glob('*.pt'):
        with p.open('rb') as f:os.fsync(f.fileno())
    files={p.name:p.stat().st_size for p in pending.glob('*.pt')}
    atomic_json(pending/'complete.json',dict(step=step,stage=config['stage'],files=files,milestone=milestone))
    result=root/name;os.replace(pending,result)
    atomic_json(root/'latest.json',dict(checkpoint=str(result.resolve()),step=step))
    complete=[p for p in root.glob('step_*') if is_complete(p)]
    ordinary=sorted((p for p in complete if not json.loads((p/'complete.json').read_text()).get('milestone')),key=lambda p:p.name)
    for old in ordinary[:-keep] if keep>0 else []:shutil.rmtree(old)
    return result
