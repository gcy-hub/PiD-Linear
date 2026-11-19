import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys

from prompts import HERE, MODELS, manifest

METRICS = ('musiq_paq2piq','niqe','deqa_score','maniqa','qalign_native','unipercept_iaa','unipercept_iqa','visualquality_r1')


def efficiency(root, rows):
    results, placement = {}, {}
    for model in MODELS:
        times = []
        for i in range(0,len(rows),2):
            r = json.loads((root/model/f'{i:04}.json').read_text())
            assert r['rows']==rows[i:i+2] and r['model']==model and not r['compile'] and r['batch_size']==2
            assert r['steps']==25 and r['width']==r['height']==4096 and (r['cfg'],r['shift'],r['sigma'])==(5.,6.,0.)
            hw = (r['host'],r['gpu'],r['gpu_uuid'],r['torch'],r['cuda'])
            if placement.setdefault(i,hw)!=hw:
                raise ValueError(f'Unmatched hardware/software for batch {i}')
            t = r['decode_seconds']
            if not math.isfinite(t) or t<=0:
                raise ValueError(f'Invalid duration: {model}/{i}')
            times.append(t)
        per_image = sum(times)/len(rows)
        results[model] = dict(batch_seconds=mean(times),seconds_per_image=per_image,
                              speedup=(results['pid']['seconds_per_image']/per_image if results else 1.))
    if len(set(placement.values())) != 8:
        raise ValueError('Expected eight distinct GPU workers')
    return dict(models=results,hardware=sorted(set(placement.values())))


def score(root, rows, metric, models=MODELS):
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0,str(HERE.parent))
    from pid._src.evaluations.metrics import MetricRegistry
    engine = MetricRegistry.get(metric,device='cuda:0',**({'backend':'transformers'} if metric=='visualquality_r1' else {}))
    output = {}
    for model in models:
        values = []
        for r in rows:
            seed = r['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            with Image.open(root/model/f'{r["id"]:04}.png') as im:
                if im.mode!='RGB' or im.size!=(4096,4096):
                    raise ValueError(f'Not a 4K RGB PNG: {im.filename}')
                with torch.inference_mode():
                    v = float(engine.compute(np.array(im)))
            if not math.isfinite(v):
                raise ValueError(f'{metric}: invalid score for {model}/{r["id"]}')
            values.append(v)
        output[model] = dict(mean=mean(values),scores_by_id=values)
    return output


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('measure',choices=('efficiency',*METRICS))
    p.add_argument('--single',action='store_true',help='Score the three PNGs for sample 0')
    args = p.parse_args()
    rows = manifest()
    if args.single:
        if args.measure == 'efficiency':
            p.error('Use single-0000.json for BS=1 decode timings')
        rows = rows[:1]
    models = (*MODELS,'pid-linear') if args.single and (args.output/'pid-linear/0000.png').is_file() else MODELS
    result = efficiency(args.output,rows) if args.measure=='efficiency' else score(args.output,rows,args.measure,models)
    target = args.output/f'{args.measure}.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(target)
