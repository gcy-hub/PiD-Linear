"""Aggregate resumable partition diagnostics; numbers describe interventions, not causal proof."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


def median(xs):return float(statistics.median(xs)) if xs else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plots',action='store_true');args=p.parse_args();root=args.output
    masks=[];single_layers=[];boundaries=[];completions=[];failures=[]
    for file in sorted(root.rglob('target_*.json')):
        value=json.loads(file.read_text());parts=file.relative_to(root).parts
        if parts[0]=='mask':
            attention={r['layer']:r for r in value['attention']}
            for row in value['single_layer']:
                single_layers.append(dict(model=parts[1],case=parts[2],target=value['target'],layer=row['layer'],
                    prediction_relative_l2=row['prediction']['relative_l2'],
                    image_attention_relative_l2=attention[row['layer']]['image']['relative_l2'],
                    text_attention_relative_l2=attention[row['layer']]['text']['relative_l2']))
            for row in value['cumulative']:
                masks.append(dict(model=parts[1],case=parts[2],target=value['target'],actual_t=value['actual_t'],
                    layers=row['count'],relative_l2=row['prediction']['relative_l2'],
                    guided_relative_l2=row['guided_prediction']['relative_l2']))
        elif parts[0]=='boundary':
            for layer in value['layers']:
                teacher=layer['teacher_discontinuity']
                for variant in layer['variants']:
                    remote=variant['remote_discontinuity'];sy,sx=variant['split']
                    xp,yp=np.array(remote['vertical_profile']),np.array(remote['horizontal_profile'])
                    def line_ratio(profile,cut):
                        keep=np.ones(len(profile),bool);keep[max(0,cut-3):min(len(profile),cut+2)]=False
                        return float(profile[cut-1]/max(profile[keep].mean(),1e-30))
                    boundaries.append(dict(model=parts[1],checkpoint=parts[2],case=parts[3],target=value['target'],
                        actual_t=value['actual_t'],layer=layer['layer'],split_y=sy,split_x=sx,
                        is_center=variant['split']==layer['variants'][0]['split'],
                        image_relative_l2=variant['image']['relative_l2'],remote_rms=variant['remote_rms'],
                        error_boundary_ratio=variant['error_spatial']['boundary_to_interior'],
                        remote_vertical_at_split=line_ratio(xp,sx),remote_horizontal_at_split=line_ratio(yp,sy),
                        remote_vertical_old_center=remote['vertical_ratio'],remote_horizontal_old_center=remote['horizontal_ratio'],
                        local_vertical_center=layer['local_discontinuity']['vertical_ratio'],
                        local_horizontal_center=layer['local_discontinuity']['horizontal_ratio'],
                        teacher_vertical_center=teacher['vertical_ratio'],teacher_horizontal_center=teacher['horizontal_ratio'],
                        remote_change_relative_l2=variant['remote_delta']['relative_l2']))
    for file in root.rglob('complete.json'):completions.append(str(file.parent.relative_to(root)))
    for file in root.rglob('failure.json'):
        if not (file.parent/'complete.json').exists():failures.append(dict(path=str(file.relative_to(root)),**json.loads(file.read_text())))
    for name,rows in [('teacher-mask.csv',masks),('teacher-single-layer.csv',single_layers),('qdm-boundaries.csv',boundaries)]:
        if rows:
            with (root/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mask_summary=[]
    for model in sorted({r['model'] for r in masks}):
        for target in sorted({r['target'] for r in masks},reverse=True):
            for layers in (1,4,8,14):
                group=[r for r in masks if (r['model'],r['target'],r['layers'])==(model,target,layers)]
                if group:mask_summary.append(dict(model=model,target=target,layers=layers,n=len(group),
                    median_relative_l2=median([r['relative_l2'] for r in group]),
                    median_guided_relative_l2=median([r['guided_relative_l2'] for r in group])))
    boundary_summary=[]
    for model,label in sorted({(r['model'],r['checkpoint']) for r in boundaries}):
        for shift in ('center','shifted'):
            group=[r for r in boundaries if (r['model'],r['checkpoint'])==(model,label) and
                   r['is_center']==(shift=='center')]
            if group:boundary_summary.append(dict(model=model,checkpoint=label,split=shift,n=len(group),
                median_image_relative_l2=median([r['image_relative_l2'] for r in group]),
                median_error_boundary_ratio=median([r['error_boundary_ratio'] for r in group]),
                median_remote_vertical_at_split=median([r['remote_vertical_at_split'] for r in group]),
                median_remote_horizontal_at_split=median([r['remote_horizontal_at_split'] for r in group]),
                median_teacher_vertical_center=median([r['teacher_vertical_center'] for r in group]),
                median_teacher_horizontal_center=median([r['teacher_horizontal_center'] for r in group])))
    summary=dict(complete=completions,pending_or_failed=failures,teacher_mask=mask_summary,qdm_boundary=boundary_summary)
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# QDM 象限诊断：原始教师限制与记忆边界','',
      '这是无训练干预。PiD 使用原生 2K、25 步、CFG 5、shift 6；PixelDiT 使用原生 1K、50 步、CFG 2.75、shift 4。两者任务、权重及分辨率不同，跨模型差异不能单独证明 latent 补偿。',
      '',f'完成 {len(completions)} 个任务；未完成任务的失败记录 {len(failures)} 个。主要任务包含 3 个最近实际噪声时间的教师轨迹输入，JSON 记录 actual_t；4K 补充使用原始模型成图的固定加噪输入，不与轨迹输入混为同一统计组。','',
      '## 第一优先：原始教师跨象限 mask','',
      '图像 query 只看本象限图像与全部文本；文本 query 保持全局。保留原始 RoPE、投影和 PiT。下表为条件速度预测相对 L2（不是 loss 或生成质量分数）；括号为 CFG 后相对 L2。跨 prompt 取中位数。','',
      '| 模型 | 目标时间 | mask 层数 | 样本数 | 条件相对 L2 | CFG 相对 L2 |',
      '|---|---:|---:|---:|---:|---:|']
    for r in mask_summary:lines.append(f"| {r['model']} | {r['target']} | {r['layers']} | {r['n']} | {r['median_relative_l2']:.4f} | {r['median_guided_relative_l2']:.4f} |")
    lines+=['','逐层 Attention 误差与单层最终预测误差在各 target JSON 中；mask_4.png、mask_14.png 使用完整原生采样。','',
      '## 第二优先：已训练 QDM 的象限边界','',
      '在教师相同层输入上计算 local、remote 与融合输出，只移动 remote 的 read/write 分界，原始坐标与局部 Halo 不变。center 为原划分；shifted 为 3/8、5/8 的分界。边界梯度比 = 分界相邻特征差异能量 / 其他位置均值，1 表示接近背景。跨层、噪声与 prompt 的中位数仅用于概览；逐层结果请看 CSV。','',
      '| 模型 / checkpoint | 划分 | Attention 相对 L2 | 误差边界比 | remote 竖边界比 | remote 横边界比 |',
      '|---|---|---:|---:|---:|---:|']
    for r in boundary_summary:lines.append(f"| {r['model']} / {r['checkpoint']} | {r['split']} | {r['median_image_relative_l2']:.4f} | {r['median_error_boundary_ratio']:.3f} | {r['median_remote_vertical_at_split']:.2f} | {r['median_remote_horizontal_at_split']:.2f} |")
    lines+=['','远程特征中的接缝不等于最终图像一定有可见接缝：local、后续层与 PiT 可能补偿。平移分界是推理分布干预，不能直接据此比较架构生成质量。不同 checkpoint 训练阶段与预算不同，不能把两者误差差异当作受控架构对照。',
      '', '## 记录与复现','',
      '- `teacher-mask.csv`：逐 prompt、噪声与 mask 层数的整网结果。',
      '- `teacher-single-layer.csv`：逐层 Attention 近似误差及该层干预的最终预测误差。',
      '- `qdm-boundaries.csv`：逐层原始及平移分界统计，包含 local、教师控制值。',
      '- `*/target_*.json`：完整逐层、逐象限能量、空间剖面和输入时间。',
      '- `*/target_*_maps.npz`：原尺寸 token 误差和 remote 能量图。',
      '- `*/config.json`：checkpoint、设置、prompt、seed。',
      '- `*/complete.json`：该任务完成标记、实际峰值显存与总耗时。']
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image,ImageDraw
        for case_dir in root.glob('mask/*/*'):
            paths=[case_dir/name for name in ('original.png','mask_4.png','mask_14.png')]
            if all(p.exists() for p in paths):
                images=[Image.open(p).convert('RGB') for p in paths];w,h=images[0].size
                canvas=Image.new('RGB',(w*3,h+40),'white');draw=ImageDraw.Draw(canvas)
                for i,(im,label) in enumerate(zip(images,['Original Full Attention','Mask 4 layers','Mask all 14 layers'])):
                    canvas.paste(im,(w*i,40));draw.text((w*i+20,12),label,fill='black')
                canvas.save(case_dir/'comparison.png')
        for file in root.glob('boundary/*/*/*/target_500_maps.npz'):
            with np.load(file) as maps:
                fig,axes=plt.subplots(3,3,figsize=(12,11))
                for i,layer in enumerate((0,7,13)):
                    arrays=[maps[f'l{layer:02d}_split{v}_error'] for v in range(3)]
                    limit=max(np.quantile(a,.99) for a in arrays)
                    for v,a in enumerate(arrays):
                        ax=axes[i,v];ax.imshow(a,cmap='magma',vmin=0,vmax=limit)
                        h,w=a.shape;f=(.5,.375,.625)[v]
                        ax.axhline(int(h*f)-.5,color='cyan',lw=.8);ax.axvline(int(w*f)-.5,color='cyan',lw=.8)
                        ax.set_title(f'Layer {layer}, cut {f:.3f}');ax.set_xticks([]);ax.set_yticks([])
                fig.suptitle('Teacher-input attention squared error; same scale within each row')
                fig.tight_layout();fig.savefig(file.parent/'boundary-errors.png',dpi=140);plt.close(fig)
    lines+=['','## 原始教师成图对比','']
    for im in sorted(root.glob('mask/*/*/comparison.png')):
        lines+=['### '+str(im.parent.relative_to(root)), '', '从左到右：原始 Full Attention、4 层象限 mask、14 层象限 mask。', '', '![]('+str(im.relative_to(root))+')', '']
    lines+=['## QDM 边界误差图','']
    for im in sorted(root.glob('boundary/*/*/*/boundary-errors.png')):
        lines+=['### '+str(im.parent.relative_to(root)), '', '![]('+str(im.relative_to(root))+')', '']
    if args.plots:
        for file in root.glob('boundary/*/*/*/target_500.json'):
            value=json.loads(file.read_text());layer=value['layers'][0]
            fig,axes=plt.subplots(3,2,figsize=(11,8))
            for i,variant in enumerate(layer['variants']):
                for axis,(profile,cut,ref_key) in enumerate((
                    ('vertical_profile',variant['split'][1],'vertical_control'),
                    ('horizontal_profile',variant['split'][0],'horizontal_control'))):
                    ax=axes[i,axis];remote=variant['remote_discontinuity']
                    z=np.array(remote[profile]);position=(np.arange(len(z))+1)/(len(z)+1)
                    ax.plot(position,z/max(remote[ref_key],1e-30),label='Remote QDM')
                    local=layer['local_discontinuity'];teacher=layer['teacher_discontinuity']
                    ax.plot(position,np.array(local[profile])/max(local[ref_key],1e-30),alpha=.6,label='Local Halo')
                    ax.plot(position,np.array(teacher[profile])/max(teacher[ref_key],1e-30),alpha=.6,label='Teacher')
                    ax.axvline(cut/(len(z)+1),c='red',ls='--',lw=.8)
                    ax.axvline(.5,c='gray',ls=':',lw=.8)
                    ax.set_title(f'Layer 0, {profile}, split {cut/(len(z)+1):.3f}')
                    ax.set_ylabel('Adjacent feature difference / control')
                    if i==0 and axis==0:ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(file.parent/'boundary-profiles.png',dpi=140);plt.close(fig)
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=len(completions),failures=len(failures),mask_rows=len(masks),boundary_rows=len(boundaries))))


if __name__=='__main__':main()
