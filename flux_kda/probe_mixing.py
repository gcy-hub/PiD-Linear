"""Measure global/local text mass and KDA scale on fixed FLUX3 inputs."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import types
import torch
from .model import build_model, load_delta, teacher_mode
from .attention import ImageKDA
from .data import ClearFlux3Dataset
from flux3_qdm.model import native
from flux3_qdm.runtime import pack_samples


def portions(q, k, v, grid, text, kernel=11):
    height, width = grid
    n = height * width
    rows = torch.linspace(0, height-1, min(8,height),device=q.device).long()
    cols = torch.linspace(0, width-1, min(8,width),device=q.device).long()
    ids = (rows[:,None]*width+cols).flatten()
    score = (q[:,:,ids].float() @ k.float().transpose(-1,-2)) / q.shape[-1]**0.5
    keys = torch.arange(n,device=q.device)
    mask = ((ids[:,None]//width-keys//width).abs()<=kernel//2) & ((ids[:,None]%width-keys%width).abs()<=kernel//2)
    local_score = score.clone()
    local_score[:,:,:,:n].masked_fill_(~mask,float('-inf'))
    full, local = score.softmax(-1), local_score.softmax(-1)
    full_image_lse = score[:,:,:,:n].logsumexp(-1)
    local_image_lse = local_score[:,:,:,:n].logsumexp(-1)
    return ids, full, local, (local_image_lse-full_image_lse).mean((0,2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--calibration',help='Check fixed calibration on held-out samples')
    args=p.parse_args()
    torch.set_num_threads(4)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    source=Path('/home/ganchangyi/dataset/CLEAR-FLUX3-klein-base-2K-2048_20261106-1102/CLEAR-FLUX3-klein-base-2K-2048')
    data=ClearFlux3Dataset(source,source/'cache-flux3-1024',2000)
    model=build_model('flux3','/home/ganchangyi/huggingface_ckpts/flux-3-image-klein-base/model.safetensors','cuda')
    state=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    load_delta(model,state['delta']);step=state['step'];del state
    for module in model.modules():
        if isinstance(module,ImageKDA):module.inference_weights()
    for name,param in model.named_parameters():
        if name.endswith('train_weight'):param.data=param.data.to(torch.bfloat16)
    model.eval().requires_grad_(False)
    context={};records=[];kda_outputs={}
    calibration=json.loads(Path(args.calibration).read_text())['text_logit_bias'] if args.calibration else {}
    def install(block):
        original=block.forward
        def capture(module,inputs,output):kda_outputs[block.layer_index]=output[0].detach()
        block.kda.register_forward_hook(capture)
        def forward(self,x,freqs,bounds,mod,**kwargs):
            shift,scale,gate=mod
            z=native.layernorm_modulation(x,scale,shift)
            raw=self.raw_qkv(z)
            q,k=self.orig.norm(raw[0],raw[1],freqs);v=raw[2].to(q.dtype)
            sample=kwargs['samples'][0]
            n=sample.image_end-sample.image_start;text=sample.text_end-sample.text_start
            ids,full,local,bias=portions(q,k,v,sample.grid,text)
            wanted=kwargs.get('return_attention_feature',False)
            kwargs['return_attention_feature']=True
            result,feature=original(x,freqs,bounds,mod,**kwargs)
            vf=v.float()
            global_out=full@vf
            local_image=local[:,:,:,:n]@vf[:,:,:n]
            local_text=local[:,:,:,n:]@vf[:,:,n:]
            actual=feature[ids].reshape(len(ids),self.heads,self.d).permute(1,0,2)[None].float()
            def rms(value):return value.square().mean().sqrt().item()
            row=dict(context,layer=self.layer_index,text_tokens=text,
                     full_text_mass=full[:,:,:,n:].sum(-1).mean().item(),
                     local_text_mass=local[:,:,:,n:].sum(-1).mean().item(),
                     full_rms=rms(global_out),local_image_rms=rms(local_image),local_text_rms=rms(local_text),
                     local_output_relative_mse=((local_image+local_text-global_out).square().mean()/global_out.square().mean().clamp_min(1e-20)).item(),
                     output_relative_mse=((actual-global_out).square().mean()/global_out.square().mean().clamp_min(1e-20)).item(),
                     logit_bias=bias.cpu().tolist())
            if not kwargs.get('teacher_mode',False):
                image=kda_outputs.pop(self.layer_index)[ids].reshape(len(ids),self.heads,self.d).permute(1,0,2)[None].float()
                row.update(kda_rms=rms(image),kda_to_local_rms=rms(image)/max(rms(local_image+local_text),1e-20),
                           kda_residual_cosine=torch.nn.functional.cosine_similarity(image.flatten(),(global_out-local_image-local_text).flatten(),dim=0).item(),
                           kda_full_cosine=torch.nn.functional.cosine_similarity(image.flatten(),global_out.flatten(),dim=0).item())
                if str(self.layer_index) in calibration:
                    corrected=local.clone()
                    factors=torch.tensor(calibration[str(self.layer_index)],device=q.device).exp()
                    corrected[:,:,:,n:]*=factors[None,:,None,None]
                    corrected/=corrected.sum(-1,keepdim=True)
                    adjusted=corrected@vf+image
                    row.update(calibrated_text_mass=corrected[:,:,:,n:].sum(-1).mean().item(),
                               calibrated_output_relative_mse=((adjusted-global_out).square().mean()/global_out.square().mean().clamp_min(1e-20)).item())
            records.append(row)
            return (result,feature) if wanted else result
        block.forward=types.MethodType(forward,block)
    for block in model.qdm_blocks:install(block)
    with torch.inference_mode(),(root/'measurements.jsonl').open('w') as stream:
        for index in ([150,750] if calibration else [0,333,999,1666,150,750]):
            sample=data[index]
            noise=torch.randn(sample['image_latents'].shape,device='cuda',dtype=torch.bfloat16,
                              generator=torch.Generator(device='cuda').manual_seed(index))
            for sigma in [.2,.5,.8]:
                packed=pack_samples([sample],[(1-sigma)*sample['image_latents'].cuda()+sigma*noise],[sigma],'cuda')
                predictions={}
                for mode in ['teacher','student']:
                    context.update(sample=index,split='calibration' if index in [0,333,999,1666] else 'heldout',sigma=sigma,mode=mode)
                    start=len(records)
                    with teacher_mode(model) if mode=='teacher' else nullcontext():
                        predictions[mode]=model(packed).float()
                    for row in records[start:]:stream.write(json.dumps(row)+'\n')
                    stream.flush()
                error=((predictions['student']-predictions['teacher']).square().mean()/predictions['teacher'].square().mean()).item()
                print(json.dumps(dict(sample=index,sigma=sigma,prediction_relative_mse=error)),flush=True)
    if calibration:
        (root/'COMPLETE').write_text('Held-out attention measurements completed.\n')
        print('Completed held-out calibration check',flush=True)
        return
    bias={}
    for layer in range(5,33):
        rows=[r['logit_bias'] for r in records if r['mode']=='student' and r['split']=='calibration' and r['layer']==layer]
        bias[str(layer)]=torch.tensor(rows).mean(0).tolist()
    (root/'calibration.json').write_text(json.dumps(dict(checkpoint_step=step,text_logit_bias=bias),indent=2)+'\n')
    print('Completed measurements and calibration',flush=True)


if __name__=='__main__':main()
