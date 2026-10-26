"""Common-input error propagation and original-Attention recovery; no optimization."""
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from tqdm import tqdm

from .run_partition_diagnostics import load_student, call, write_json, save_image
from pid._src.vdn_pid.attention import execution_mode


def metrics(value, reference):
    x, y = value.detach().float().flatten(), reference.detach().float().flatten()
    delta = x-y
    return dict(relative_l2=float(delta.norm()/y.norm().clamp_min(1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(x[None],y[None]).item()),
                output_rms=float(x.square().mean().sqrt()),
                reference_rms=float(y.square().mean().sqrt()),
                max_abs=float(delta.abs().max()))


def pair_metrics(value, reference):
    return {stream:metrics(x,y) for stream,x,y in zip(('image','text'),value,reference)}


@contextmanager
def restored(net, indices=(), teacher=False):
    with execution_mode(net,teacher=teacher,inference=True):
        for i in indices:
            net.patch_blocks[i].attn.teacher_mode=True
        yield


def followup_layouts():
    return {'single5':[5], 'middle_without5':[6,7,8], 'worst_without5':[2,3,4]}


def layouts(worst=None):
    result={'qdm':[], 'front4':[0,1,2,3], 'middle4':[5,6,7,8],
            'last4':[10,11,12,13], 'spread4':[0,4,8,12], 'alternating7':list(range(0,14,2))}
    if worst is not None:result['worst4']=sorted(worst)
    return result


class LayerProbe:
    """Replay the same teacher Block input without affecting the teacher main path."""
    def __init__(self,net):
        self.net=net;self.mode=None;self.guard=False;self.refs={};self.rows={};self.handles=[]
        for i,block in enumerate(net.patch_blocks):
            def attention(m,args,kwargs,out,i=i):
                if self.guard or self.mode is None:return
                if self.mode=='teacher':
                    self.refs[i,'attn']=tuple(x.detach() for x in out)
                    self.guard=True
                    try:
                        student=m.student_forward(*args,**kwargs)
                    finally:self.guard=False
                    self.rows[i]={'layer':i,'teacher_forced_attention':pair_metrics(student,out)}
                elif self.mode=='student':
                    self.rows[i]['rollout_attention']=pair_metrics(out,self.refs[i,'attn'])
            def complete(m,args,kwargs,out,i=i):
                if self.guard or self.mode is None:return
                if self.mode=='teacher':
                    self.refs[i,'block']=tuple(x.detach() for x in out)
                    self.guard=True;old=m.attn.teacher_mode;m.attn.teacher_mode=False
                    try:student=m.forward(*args,**kwargs)
                    finally:m.attn.teacher_mode=old;self.guard=False
                    self.rows[i]['teacher_forced_block']=pair_metrics(student,out)
                elif self.mode=='student':
                    self.rows[i]['rollout_block']=pair_metrics(out,self.refs[i,'block'])
            self.handles.append(block.attn.register_forward_hook(attention,with_kwargs=True))
            self.handles.append(block.register_forward_hook(complete,with_kwargs=True))
    def close(self):
        self.mode=None
        for h in self.handles:h.remove()


def asset(args,case):
    from .conditioning import load_null_cache
    a=torch.load(args.assets/(case+'.pt'),map_location='cpu',weights_only=False)
    special=load_null_cache(args.text_cache)
    a['null_embs'],a['null_mask']=special['null_embs'],special['null_mask']
    a['height']=max(16,round(a['height']/2/16)*16)
    a['width']=max(16,round(a['width']/2/16)*16)
    a.pop('latent',None)
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in a.items()}


def clean_image(a):
    with Image.open(a['source_image']) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        w,h=a['width'],a['height'];scale=max(w/im.width,h/im.height)
        im=im.resize((max(w,round(im.width*scale)),max(h,round(im.height*scale))),Image.Resampling.LANCZOS)
        left,top=(im.width-w)//2,(im.height-h)//2
        im=im.crop((left,top,left+w,top+h))
        return torch.from_numpy(np.array(im)).permute(2,0,1)[None].cuda().float()/127.5-1


@torch.no_grad()
def diagnose(net,a,times,dest,check=False):
    clean=clean_image(a)
    noise=torch.randn(clean.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(a['seed']))
    probe=LayerProbe(net)
    try:
        for t in times:
            path=dest/f't_{t:04d}.json'
            if path.exists():continue
            x=(1-t/1000)*clean+(t/1000)*noise
            ts=torch.tensor([float(t)],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                probe.mode='teacher'
                with restored(net,teacher=True):teacher=call(net,a,x,ts,kind='pixeldit')
                probe.mode='student'
                with restored(net):student=call(net,a,x,ts,kind='pixeldit')
                probe.mode=None
                out=dict(time=t,actual_bf16_time=float(ts.bfloat16()[0]),prediction=metrics(student,teacher),
                         layers=[probe.rows[i] for i in range(len(net.patch_blocks))],recovery={})
                if check:
                    wrappers=[b.attn for b in net.patch_blocks]
                    try:
                        for b,m in zip(net.patch_blocks,wrappers):b.attn=m.orig
                        original=call(net,a,x,ts,kind='pixeldit')
                    finally:
                        for b,m in zip(net.patch_blocks,wrappers):b.attn=m
                    out['wrapper_vs_native']=metrics(teacher,original)
                    compact=net.pixel_embedder.compact_image_positions
                    try:
                        net.pixel_embedder.compact_image_positions=False
                        with restored(net,teacher=True):dense_positions=call(net,a,x,ts,kind='pixeldit')
                    finally:net.pixel_embedder.compact_image_positions=compact
                    out['compact_vs_dense_rope']=metrics(teacher,dense_positions)
                    if out['wrapper_vs_native']['max_abs']!=0 or out['compact_vs_dense_rope']['max_abs']!=0:
                        raise RuntimeError('P0 wrapper/RoPE equivalence failed; inspect before recovery experiments')
                for name,indices in layouts().items():
                    if name=='qdm':value=student
                    else:
                        with restored(net,indices):value=call(net,a,x,ts,kind='pixeldit')
                    out['recovery'][name]=metrics(value,teacher)
                write_json(path,out)
                probe.refs.clear();probe.rows.clear()
            print(f'FINISHED {dest.name} t={t} pred_rel={out["prediction"]["relative_l2"]:.5f}',flush=True)
    finally:probe.close()


@torch.no_grad()
def selected_recovery(net,args,rank,world):
    selection=json.loads((args.output/'summary.json').read_text())['worst4']
    cases=sorted(args.assets.glob('real_*_2048.pt'))[:args.samples]
    for index,path in enumerate(cases):
        if index%world!=rank:continue
        a=asset(args,path.stem);clean=clean_image(a)
        noise=torch.randn(clean.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(a['seed']))
        for t in args.times:
            dest=args.output/'errors'/path.stem/f't_{t:04d}.json'
            row=json.loads(dest.read_text())
            if 'worst4' in row['recovery']:continue
            x=(1-t/1000)*clean+(t/1000)*noise;ts=torch.tensor([float(t)],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                with restored(net,teacher=True):teacher=call(net,a,x,ts,kind='pixeldit')
                with restored(net,selection):value=call(net,a,x,ts,kind='pixeldit')
            row['recovery']['worst4']=metrics(value,teacher)
            row['worst4_layers']=selection
            write_json(dest,row)
        print(f'RECOVERY {path.stem} worst4={selection}',flush=True)


def aggregate(args):
    rows=[json.loads(p.read_text()) for p in sorted((args.output/'errors').glob('*/t_*.json'))]
    if not rows:raise RuntimeError('No completed diagnostic inputs')
    bylayer=[]
    for i in range(14):
        item={'layer':i}
        for mode in ('teacher_forced_attention','teacher_forced_block','rollout_attention','rollout_block'):
            for stream in ('image','text'):
                for metric in ('relative_l2','cosine','output_rms','reference_rms'):
                    item[f'{mode}/{stream}/{metric}']=float(np.mean([r['layers'][i][mode][stream][metric] for r in rows]))
        bylayer.append(item)
    # Rank pre-intervention teacher-forced image Attention errors; not chosen by generated image quality.
    worst=sorted(range(14),key=lambda i:bylayer[i]['teacher_forced_attention/image/relative_l2'],reverse=True)[:4]
    summary=dict(inputs=len(rows),layers=bylayer,worst4=sorted(worst),
        consistency={key:max(r[key]['max_abs'] for r in rows if key in r) for key in ('wrapper_vs_native','compact_vs_dense_rope')},
        prediction_by_time={str(t):float(np.mean([r['prediction']['relative_l2'] for r in rows if r['time']==t])) for t in args.times},
        recovery={name:float(np.mean([r['recovery'][name]['relative_l2'] for r in rows])) for name in rows[0]['recovery'] if all(name in r['recovery'] for r in rows)},
        per_time={str(t):{key:float(np.mean([r['layers'][i][key]['image']['relative_l2'] for r in rows if r['time']==t for i in range(14)]))
                         for key in ('teacher_forced_attention','rollout_attention','teacher_forced_block','rollout_block')} for t in args.times})
    with (args.output/'layer_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bylayer[0]));writer.writeheader();writer.writerows(bylayer)
    write_json(args.output/'summary.json',summary)
    return summary


@torch.no_grad()
def render(net,args,rank,world,selection):
    from .runtime import sample
    variants=followup_layouts() if args.phase=='followup' else {'teacher':list(range(14)),**layouts(selection)}
    jobs=[(case,name,indices) for case in args.generate_cases for name,indices in variants.items()]
    for case,name,indices in tqdm(jobs[rank::world],desc=f'Generation GPU {rank}'):
        dest=args.output/'generation'/case/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'done.json').exists():continue
        a=asset(args,case)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with restored(net,indices):
            image=sample(net,a['caption_embs'],a['caption_mask'],a['null_embs'],a['null_mask'],
                height=a['height'],width=a['width'],seed=a['seed'],steps=50,cfg=2.75,shift=4)
        torch.cuda.synchronize();save_image(dest/'image.png',image)
        write_json(dest/'done.json',dict(case=case,restored=indices,seed=a['seed'],prompt=a['caption'],
            height=a['height'],width=a['width'],steps=50,cfg=2.75,shift=4,seconds=time.perf_counter()-started,
            peak_gib=torch.cuda.max_memory_allocated()/2**30))
        print(f'GENERATED {case} {name}',flush=True)


def report(args):
    summary=aggregate(args)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,part in zip(axs,('attention','block')):
        for mode in ('teacher_forced','rollout'):
            ax.plot(range(14),[r[f'{mode}_{part}/image/relative_l2'] for r in summary['layers']],marker='o',label=mode)
        ax.set(xlabel='MMDiT layer (zero based)',ylabel='Relative L2',title=part);ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(args.output/'layer_errors.png',dpi=160);plt.close(fig)
    text=['# PixelDiT-T2I：QDM 恢复诊断','',
        '固定共同输入，对比逐层 Teacher-forced 与 Student rollout；不更新模型参数。',
        f'Checkpoint：`{args.checkpoint}`。使用原始图像加噪、原生约 1K 多长宽比；时间尺度 0–1000，950 为高噪声。',
        f'完成 {summary["inputs"]} 个样本×时间点。Attention 指标在输出投影后计算，Block 指标在两次残差及 FFN 之后计算。',
        '图像、文本分别统计。恢复层使用原始权重的完整 Full Attention，其余学生模块保持不变。',
        '', '![逐层误差](layer_errors.png)','', '| 恢复方案 | 层索引 | 最终预测相对 L2（平均） |','|---|---|---:|']
    for name,value in summary['recovery'].items():text.append(f'| {name} | {layouts(summary["worst4"])[name]} | {value:.6f} |')
    base=summary['recovery']['qdm']
    best=min(summary['recovery'],key=summary['recovery'].get)
    text+=['','## 数值诊断结果',
        f"教师封装与原始 Attention 的最大绝对差：{summary['consistency']['wrapper_vs_native']}；压缩与完整 RoPE 坐标的最大绝对差：{summary['consistency']['compact_vs_dense_rope']}。",
        f"共同输入下，全部 QDM 的最终预测平均相对 L2 为 {base:.4%}；本轮恢复方案中最小的是 {best}（{summary['recovery'][best]:.4%}）。这只是单步预测误差排序，需要再对照生成图。",
        '', '| 噪声时间 | Teacher-forced Attention（图像） | Rollout Attention（图像） | Teacher-forced Block（图像） | Rollout Block（图像） | 最终预测 |',
        '|---|---:|---:|---:|---:|---:|']
    for t in args.times:
        r=summary['per_time'][str(t)]
        text.append('| '+str(t)+' | '+' | '.join(f'{r[k]:.2%}' for k in ('teacher_forced_attention','rollout_attention','teacher_forced_block','rollout_block'))+f" | {summary['prediction_by_time'][str(t)]:.2%} |")
    text+=['','Attention 在相同教师输入下已存在明显偏差，Rollout 的 Block 误差又进一步增大：本轮结果同时支持单层近似不足及多层传播放大，尚不能单独判定是记忆表达能力还是优化目标造成。',
        '不同噪声时间下最终预测包含的噪声/细节比例不同，不能仅凭最终相对误差排序决定哪个阶段对成图最关键。',
        '文本指标包含原生 padding 位置；生成比较使用原生 CFG 条件与负提示词两路。']
    text+=['',f'高误差四层（依据教师输入下 Attention 图像误差，非依据成图挑选）：{summary["worst4"]}。',
           '', '相对 L2 是误差范数除以教师输出范数，并非相对 MSE；Attention 和 Block 的幅度不同，数值不能直接当作对图像质量的影响。',
           '恢复层属于推理干预，会改变后续层的输入分布；需要结合相同输入的预测误差及固定种子成图判断。', '', '## 固定种子生成比较', '生成输入为 caption 的文本编码和固定像素噪声，50 步、CFG 2.75、shift 4。以下案例仅用于机制诊断，不能用四个 prompt 代表整体生成质量。']
    names=['teacher',*layouts(summary['worst4'])]
    for case in args.generate_cases:
        paths=[args.output/'generation'/case/name/'image.png' for name in names]
        if not all(p.exists() for p in paths):continue
        ims=[Image.open(p).convert('RGB') for p in paths];w,h=ims[0].size;gap=16
        canvas=Image.new('RGB',(len(ims)*(w+gap)-gap,h+36),'white');draw=ImageDraw.Draw(canvas)
        for i,(im,name) in enumerate(zip(ims,names)):
            canvas.paste(im,(i*(w+gap),36));draw.text((i*(w+gap)+8,8),name,fill='black');im.close()
        tmp=args.output/f'.{case}_comparison.{os.getpid()}.tmp.png'
        canvas.save(tmp);tmp.replace(args.output/f'{case}_comparison.png')
        text+=['',f'### {case}',f'![{case}]({case}_comparison.png)']
        follow_names=['teacher','qdm',*followup_layouts()]
        follow_paths=[args.output/'generation'/case/name/'image.png' for name in follow_names]
        if all(p.exists() for p in follow_paths):
            canvas=Image.new('RGB',(len(follow_paths)*(w+gap)-gap,h+36),'white');draw=ImageDraw.Draw(canvas)
            for i,(path,name) in enumerate(zip(follow_paths,follow_names)):
                with Image.open(path) as im:canvas.paste(im,(i*(w+gap),36))
                draw.text((i*(w+gap)+8,8),name,fill='black')
            tmp=args.output/f'.{case}_single5_comparison.{os.getpid()}.tmp.png'
            canvas.save(tmp);tmp.replace(args.output/f'{case}_single5_comparison.png')
            text+=['',f'第 5 层生成干预：![{case} single5]({case}_single5_comparison.png)']
    if (args.output/'observations.md').exists():
        text+=['',(args.output/'observations.md').read_text()]
    (args.output/'report.md').write_text('\n\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path.home()/'huggingface_ckpts')
    p.add_argument('--assets',type=Path,default=Path.home()/'code/PiD-Linear/outputs/linear-pid/assets')
    p.add_argument('--text-cache',type=Path,default=Path.home()/'dataset/MultiAspect-4K-1M/linear_pid_text_cache')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--times',type=int,nargs='+',default=[950,750,500,250,50])
    p.add_argument('--generate-cases',nargs='*',default=['generated_009_2048','real_028_2048','generated_029_2048','real_011_2048'])
    p.add_argument('--phase',choices=['errors','recovery','generate','followup','report'],default='errors')
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(args.threads);args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='report':report(args);return
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    from pid._src.vdn_pid.checkpoint import load_state
    payload=load_state(args.checkpoint)
    protocol=dict(checkpoint=str(args.checkpoint.resolve()),base_sha256=payload['metadata']['base_sha256'],
        samples=args.samples,times=args.times,assets=str(args.assets.resolve()),
        input_protocol='noised-real-reference',resolution='1024',steps=50,cfg=2.75,shift=4,
        generation_cases=args.generate_cases)
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('Diagnostic output belongs to another checkpoint/protocol')
    if rank==0:
        from pid._src.linear_pid.data import sha256_file
        actual=sha256_file(payload['config']['teacher_checkpoint'])
        if actual!=protocol['base_sha256']:raise ValueError('Native base checkpoint fingerprint mismatch')
        write_json(manifest,protocol)
    del payload
    net,identity=load_student('pixeldit',args.weights,args.checkpoint)
    if args.phase=='errors':
        cases=sorted(args.assets.glob('real_*_2048.pt'))[:args.samples]
        for index,path in enumerate(cases):
            if index%world!=rank:continue
            a=asset(args,path.stem);dest=args.output/'errors'/path.stem;dest.mkdir(parents=True,exist_ok=True)
            write_json(dest/'sample.json',dict(case=path.stem,seed=a['seed'],source_image=a['source_image'],height=a['height'],width=a['width']))
            diagnose(net,a,args.times,dest,check=True)
    elif args.phase=='recovery':selected_recovery(net,args,rank,world)
    else:
        selection=json.loads((args.output/'summary.json').read_text())['worst4']
        render(net,args,rank,world,selection)


if __name__=='__main__':main()
