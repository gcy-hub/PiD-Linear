"""SNR-aware Haar pseudo-Huber distillation of clean predictions."""
import torch


def haar(x):
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError('Haar requires BCHW with even spatial dimensions')
    a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
    c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return torch.cat(((a+b+c+d)/2, (a-b+c-d)/2,
                      (a+b-c-d)/2, (a-b-c+d)/2), dim=1)


def spatial_predictions(prediction, batch, family):
    if family == 'pid':
        return list(prediction.split(1))
    grids = batch.inputs['_grids']
    if family == 'flux3':
        bounds = batch.inputs['packed'].mode_indices['image']
        samples = [prediction[start:end] for start, end in bounds]
    elif family == 'flux2':
        samples = list(prediction.unbind(0))
    else:
        raise ValueError(f'Unknown wavelet prediction family: {family}')
    if len(samples) != len(grids):
        raise ValueError('Prediction grids do not match the batch')
    spatial = []
    for sample, (h, w) in zip(samples, grids):
        if sample.shape[0] != h*w:
            raise ValueError('Prediction token count does not match its spatial grid')
        image = sample.transpose(0, 1).reshape(1, -1, h, w)
        if family == 'flux2':
            from diffusers import Flux2KleinPipeline
            image = Flux2KleinPipeline._unpatchify_latents(image)
        spatial.append(image)
    return spatial


def prediction_loss(student, teacher, batch, family, config):
    residuals = spatial_predictions(student.float() - teacher.detach().float(), batch, family)
    times = batch.timesteps.float().flatten()
    if len(residuals) != len(times):
        raise ValueError('Each prediction requires its own timestep')
    losses = []
    for residual, time in zip(residuals, times):
        t = time.clamp(1e-6, 1-1e-6)
        snr = ((1-t)/t).square().clamp(max=config.gamma)
        threshold = config.c_min + (config.c_max-config.c_min)*(snr/config.gamma).pow(config.alpha)
        weight = t/(1-t)*snr.pow(config.beta)
        r = haar(-time*residual)
        rho = r.square() / (torch.sqrt(1+(r/threshold).square())+1)
        losses.append(weight*rho.mean())
    return torch.stack(losses).mean()
