"""Strict, disjoint per-layer warm start from fourteen validated B checkpoints."""
import argparse
import hashlib
import re
from pathlib import Path

import torch

from .checkpoint import atomic_json, is_complete, read_checkpoint, save_checkpoint
from .pair_sweep import PAIRS
from .repeat import read_json, stage_finished


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def completed_sources(group_roots):
    result={}
    for root in map(Path,group_roots):
        for pair in PAIRS:
            stage=root/pair['name']/'b'
            if not stage.exists():continue
            if pair['name'] in result:raise ValueError(f'Duplicate pair: {pair["name"]}')
            if not stage_finished(root/pair['name'],'b'):
                raise ValueError(f'Pair B validation has not completed: {stage}')
            path=Path(read_json(stage/'finished.json')['checkpoint'])
            result[pair['name']]=path if path.is_absolute() else stage/path
    if set(result)!={p['name'] for p in PAIRS}:raise ValueError('All fourteen validated pair checkpoints are required')
    return [result[p['name']] for p in PAIRS]


def merge_checkpoints(sources,destination):
    sources=list(map(Path,sources));destination=Path(destination)
    if len(sources)!=14 or len(set(p.resolve() for p in sources))!=14:
        raise ValueError('Need fourteen distinct independent checkpoints')
    layers=set();merged={};template=None;shapes=None;history=[];config=None;metadata=None
    signature_keys=('base_fingerprint','data_fingerprint','architecture','native_dimensions','training_policy')
    config_keys=('region_size','halo','lora_rank','lora_alpha','seed','caption_dropout')
    required={f'orig.{name}.lora_{letter}' for name in ('q_proj','k_proj','v_proj','attn_out') for letter in ('A','B')}
    for path in sources:
        state=read_checkpoint(path);c=state['config'];m=state['metadata']
        chosen=c.get('qdm_layers');expected=next((p for p in PAIRS if p['layers']==chosen),None)
        if expected is None or layers.intersection(chosen):raise ValueError('Invalid or overlapping two-layer layout')
        if c.get('stage')!='b' or state['progress']['stage_step']!=50 or c.get('settings',{}).get('steps')!=50:
            raise ValueError('Sources must be the completed 50-step pair B experiments')
        if c.get('attention_mode','hybrid')!='hybrid' or c.get('memory_text_write',False) or c.get('retain_lora',False):
            raise ValueError('Pair sources must use image-only Hybrid memory without inherited LoRA')
        if c.get('halo')!=2 or c.get('region_size')!=8:raise ValueError('Expected region8 + Halo2')
        signature=({k:m.get(k) for k in signature_keys},{k:c.get(k) for k in config_keys})
        if template is None:template=signature;config=dict(c);metadata=dict(m)
        elif template!=signature:raise ValueError('Pair base, data, policy or architecture differs')
        delta=torch.load(path/'delta.pt',map_location='cpu',weights_only=True)
        tails={i:{} for i in chosen}
        for key,value in delta.items():
            match=re.fullmatch(r'base\.single_blocks\.(\d+)\.(.+)',key)
            if match is None or int(match[1]) not in chosen:raise ValueError(f'Out-of-layout delta key {key}')
            tail=match[2]
            if not (tail.startswith(('remote.','softmax_gate.')) or tail in required):
                raise ValueError(f'Unexpected base parameter {key}')
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite or non-FP32 delta: {key}')
            tails[int(match[1])][tail]=tuple(value.shape)
            if key in merged:raise ValueError(f'Duplicate delta {key}')
            merged[key]=value
        for i,shape in tails.items():
            if not required<=set(shape) or not any(k.startswith('remote.') for k in shape) or not any(k.startswith('softmax_gate.') for k in shape):
                raise ValueError(f'Incomplete Hybrid/LoRA state at layer {i}')
            if shapes is None:shapes=shape
            elif shapes!=shape:raise ValueError(f'Layer delta keys/shapes differ at layer {i}')
        layers.update(chosen)
        history.append(dict(layers=chosen,checkpoint=str(path.resolve()),delta_sha256=sha256(path/'delta.pt'),
                            stage_steps=state['progress']['stage_step'],total_steps=state['progress']['total_step']))
    if layers!=set(range(28)):raise ValueError('Merged model must cover exactly all 28 layers')
    # Provenance is independent of source order and survives restart without rewriting weights.
    history.sort(key=lambda x:x['layers'][0])
    ready=read_json(destination/'ready.json')
    if ready:
        saved=Path(ready['checkpoint'])
        if ready.get('sources')!=history or not is_complete(saved):raise ValueError('Existing merged artifact differs or is incomplete')
        if ready.get('delta_sha256')!=sha256(saved/'delta.pt'):raise ValueError('Merged artifact hash mismatch')
        return saved
    config.update(stage='merged',qdm_layers=list(range(28)),retain_lora=True,init_from='',resume='none')
    metadata.pop('qdm_layers',None);metadata.pop('swanlab_run_id',None)
    metadata.update(retain_lora=True,merge_sources=history)
    # A warm start has no optimizer/RNG/sampler to resume. Joint-stage counters start at zero.
    progress=dict(stage_step=0,total_step=0,cursor=0,samples_seen=0)
    result=save_checkpoint(destination,merged,{}, {},progress,config,metadata,rng=[],milestone=True)
    atomic_json(destination/'config.json',config)
    atomic_json(destination/'ready.json',dict(checkpoint=str(result),sources=history,delta_sha256=sha256(result/'delta.pt')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-roots',nargs='+',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    print(merge_checkpoints(completed_sources(args.group_roots),args.output_root))


if __name__=='__main__':main()
