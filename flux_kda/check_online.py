"""CPU contracts for online data, dynamic batches and interrupted validation recovery."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from PIL import Image
from types import SimpleNamespace
from .online import OnlineDataset, OnlineEncoder
from .train import parse_args
from . import run
from flux3_qdm.runtime import GlobalSampler


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows=[]
        for i,size in enumerate(((4096,4096),(2200,2200),(1024,1024))):
            path=root/f'{i}.png';Image.new('RGB',size).save(path)
            rows.append(dict(id=str(i),image_path=path.name,caption='test'))
        (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        dataset=OnlineDataset(root)
        assert len(dataset)==2 and dataset.rejected==1
        assert [r['size'] for r in dataset.rows]==[(4096,4096),(2048,2048)]
        encoder = OnlineEncoder.__new__(OnlineEncoder)
        encoder.device = 'cpu'
        encoder.image = SimpleNamespace(vae=SimpleNamespace(encode=lambda x: x[:,:1,::16,::16].expand(-1,128,-1,-1)))
        encoder.text = None
        encoder.prompt = SimpleNamespace(process=lambda pipe, caption: dict(prompt_tokens=torch.zeros(80,20480,dtype=torch.bfloat16),prompt_ids=torch.zeros(80,4)))
        encoded = encoder([dataset[1]])[0]
        assert encoded['grid'] == (128,128) and encoded['image_latents'].shape == (16384,128)
        assert encoded['image_ids'].shape == (16384,4)
        assert not encoded['image_latents'].is_inference()
        projection = torch.nn.Linear(128,1,dtype=torch.bfloat16)
        projection(encoded['image_latents']).float().square().mean().backward()
        assert torch.isfinite(projection.weight.grad).all()
        sampler=GlobalSampler(dataset.bucket_indices,4,8,1)
        assert sampler.effective==32 and len(sampler.global_batch(0))==32
        assert sampler.global_batch(0)==sampler.global_batch(0)
        assert sum(len(sampler.rank_batch(0,r)) for r in range(8))==32
        small=GlobalSampler({'one':[0]},2,1,3)
        assert small.global_batch(0)==[0]*6
        output=root/'output';output.mkdir()
        config=root/'config.json'
        config.write_text(json.dumps(dict(weights='unused',subset_root=str(root),output=str(output),
                                         gpus=1,steps=3,save_every=2,validation={'size':4096})))
        args=parse_args(['--config',str(config),'--batch-size','7'])
        assert args.batch_size==7 and args.data_mode=='online' and not args.shard_optimizer
        assert not args.gradient_checkpointing and not args.optimizer_cpu_offload
        calls=[];fail=[True]
        def execute(command,**kwargs):
            if 'flux_kda.train' in command:
                target=int(command[command.index('--steps')+1]);calls.append(('train',target))
                torch.save({'step':target},output/'latest.pt')
            else:
                directory=Path(command[command.index('--output')+1]);step=int(directory.name.split('-')[1])
                calls.append(('validate',step))
                assert '--no-condition-cache' in command
                if fail[0]:
                    fail[0]=False
                    raise RuntimeError('simulated validation interruption')
                directory.mkdir(parents=True,exist_ok=True);(directory/'kda.json').write_text('{}')
        with patch('sys.argv',['run','--config',str(config)]), patch.object(run.subprocess,'run',side_effect=execute):
            try:run.main()
            except RuntimeError as exc:assert 'simulated' in str(exc)
            else:raise AssertionError('Expected validation interruption')
            run.main()
            assert calls==[('train',2),('validate',2),('validate',2),('train',3),('validate',3)]
            run.main()
            assert len(calls)==5
        assert list(output.glob('*.pt'))==[output/'latest.pt']
    print('PASS: official buckets, detected counts, small buckets, global batch, config overrides, validation retry and latest-only checkpoint lifecycle')


if __name__=='__main__':main()
