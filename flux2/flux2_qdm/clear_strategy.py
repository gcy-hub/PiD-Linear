"""CLEAR's direct-projection, three-loss, Prodigy training policy."""
import torch

COMPUTE=dict(schema=2,autocast_dtype=None,frozen_weights='bfloat16',trainable_weights='float32_master',projection_compute_weights='bfloat16',
             optimizer='prodigyopt==1.1.2',objective='fm+0.5*prediction_kd+0.5*selected_attention_kd',
             scheduler='constant',warmup=0)


def compute_context(device_type='cuda'):
    # The reference DeepSpeed BF16 engine has native_amp=False. Both training
    # and standalone inference therefore execute the native BF16 graph.
    return torch.autocast(device_type,enabled=False)


def make_optimizer(parameters, shard=False, optimizer_class=None):
    from prodigyopt import Prodigy
    optimizer_class = optimizer_class or Prodigy
    parameters=list(parameters)
    if any(p.dtype not in (torch.float32, torch.bfloat16) for p in parameters):raise ValueError('Prodigy parameters must be FP32 or BF16')
    options=dict(lr=1.,betas=(.9,.999),beta3=None,eps=1e-8,
                 weight_decay=1e-4,decouple=True,use_bias_correction=True,
                 safeguard_warmup=True,fsdp_in_use=shard)
    if shard:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer=ZeroRedundancyOptimizer(parameters,optimizer_class=optimizer_class,**options)
    else:
        optimizer=optimizer_class(parameters,**options)
    # Prodigy's global statistics must include every optimizer shard.
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)


def objective(student,teacher,target,student_features,teacher_features):
    if not student_features or len(student_features)!=len(teacher_features):
        raise ValueError('Missing or mismatched CLEAR feature layers')
    if student.shape!=teacher.shape or student.shape!=target.shape:raise ValueError('Prediction layout mismatch')
    terms=[]
    for s,t in zip(student_features,teacher_features):
        if s.shape!=t.shape:raise ValueError('Attention feature layout mismatch')
        terms.append(torch.nn.functional.mse_loss(s.float(),t.detach().float()))
    fm=torch.nn.functional.mse_loss(student.float(),target.float())
    kd=torch.nn.functional.mse_loss(student.float(),teacher.detach().float())
    feature=torch.stack(terms).mean()
    return dict(loss=fm+.5*kd+.5*feature,loss_fm=fm,loss_distill=kd,loss_attn=feature)
