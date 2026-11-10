"""CLEAR prediction and selected attention feature distillation."""
import torch
from torch.nn.functional import mse_loss

ROLES = {'branch', 'pooling', 'local_gate', 'image_qkvo'}


def objective(student, teacher, target, layers, weights):
    if not layers or any(k not in student.features or k not in teacher.features for k in layers):
        raise ValueError('Missing CLEAR feature layers')
    feature = torch.stack([mse_loss(student.features[k].float(), teacher.features[k].detach().float()) for k in layers]).mean()
    return (weights.fm * mse_loss(student.prediction.float(), target.float())
            + weights.prediction * mse_loss(student.prediction.float(), teacher.prediction.detach().float())
            + weights.feature * feature)
