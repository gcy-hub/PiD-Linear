"""Describe paired original-resolution RGB differences, without treating content changes as color proof."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageFilter


def analyze_pair(row,root):
    paths=[Path(row[k]) for k in ('original','student')]
    if not all(p.exists() for p in paths):return None
    original,student=[Image.open(p).convert('RGB') for p in paths]
    if original.size!=student.size:raise ValueError('Paired image dimensions differ')
    w,h=original.size
    radius=max(h,w)/64
    # Blur each image at native dimensions; no resizing or re-encoding.
    a=np.asarray(original.filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)/255
    b=np.asarray(student.filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)/255
    delta=b-a;sy,sx=(h+1)//2,(w+1)//2
    quadrant=[delta[ys,xs].mean((0,1)).tolist() for ys,xs in (
        (slice(0,sy),slice(0,sx)),(slice(0,sy),slice(sx,w)),
        (slice(sy,h),slice(0,sx)),(slice(sy,h),slice(sx,w)))]
    result=dict(**row,width=w,height=h,gaussian_radius=radius,rgb_mean_difference=delta.mean((0,1)).tolist(),
                quadrant_rgb_mean_difference=quadrant,low_frequency_rgb_rmse=float(np.mean(delta**2)**.5),
                quadrant_bias_std=float(np.array(quadrant).std(axis=0).mean()))
    directory=root/'image-colour';directory.mkdir(exist_ok=True)
    # Figures are created later serially; no plotting state is used in workers.
    result['row_profile_rgb']=delta.mean(1)[::max(1,h//256)].tolist()
    result['column_profile_rgb']=delta.mean(0)[::max(1,w//256)].tolist()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);args=p.parse_args();root=args.output
    pairs=json.loads((root/'image-pairs.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=[r for r in pool.map(lambda row:analyze_pair(row,root),pairs) if r is not None]
    (root/'image-colour.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    if not rows:
        print('No complete image pairs found.');return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(rows),2,figsize=(10,max(3,len(rows)*2.5)),squeeze=False)
    for i,row in enumerate(rows):
        for ax,key in zip(axes[i],('column_profile_rgb','row_profile_rgb')):
            values=np.array(row[key]);position=np.linspace(0,1,len(values))
            for c,color in enumerate(('r','g','b')):ax.plot(position,values[:,c],color=color,lw=.8)
            ax.axvline(.5,c='black',ls='--',lw=.8);ax.axhline(0,c='gray',lw=.5)
            ax.set_title(f"{row['case']} {row['resolution']} {key.split('_')[0]}")
            ax.set_ylabel('Student - teacher RGB (0..1)')
    fig.tight_layout();fig.savefig(root/'image-colour/profiles.png',dpi=140);plt.close(fig)
    print(f'Analyzed {len(rows)} existing matched images; no new image synthesis.')


if __name__=='__main__':main()
