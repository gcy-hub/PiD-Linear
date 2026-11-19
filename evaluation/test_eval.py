import json

import pytest
import torch

from prompts import manifest, pairs
from report import MODELS, efficiency


def test_shards():
    rows = manifest()
    shards = [pairs(rank) for rank in range(8)]
    assert sorted(len(s) for s in shards)==[62]*4+[63]*4
    assert sorted(i for s in shards for pair in s for i in pair)==list(range(1000))
    assert len({r['source_id'] for r in rows})==1000


def test_efficiency(tmp_path):
    rows = manifest()[:16]
    for n,model in enumerate(MODELS):
        folder = tmp_path/model
        folder.mkdir()
        for i in range(0,16,2):
            record = dict(model=model,rows=rows[i:i+2],compile=False,batch_size=2,steps=25,width=4096,height=4096,
                          cfg=5.,shift=6.,sigma=0.,host='node',gpu='A40',gpu_uuid=str(i),torch='test',cuda='test',decode_seconds=6/(n+1))
            path = folder/f'{i:04}.json'
            path.write_text(json.dumps(record))
    results = efficiency(tmp_path,rows)['models']
    assert tuple(results) == ('pid', 'pid-dit')
    assert results['pid-dit'] == dict(batch_seconds=3, seconds_per_image=1.5, speedup=2)
    record['gpu_uuid']='different'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='hardware'):
        efficiency(tmp_path,rows)


def test_two_image_cfg():
    from pid._src.common.runtime import sample
    pos = torch.tensor([[[1.],[2.]],[[3.],[4.]]])
    mask = torch.tensor([[True,False],[True,True]])
    neg = torch.zeros_like(pos[:1])
    seen = set()
    def net(x,t,y,*,lq_latent,degrade_sigma,text_valid_mask):
        assert x.shape==(2,3,16,16) and lq_latent.shape[0]==2
        if y.any():
            assert torch.equal(y,pos) and torch.equal(text_valid_mask,mask)
            seen.add('pos')
        else:
            assert text_valid_mask.shape==(2,2)
            seen.add('neg')
        return torch.zeros_like(x)
    image = sample(net,pos,mask,neg,mask[:1],torch.zeros(2,16,1,1),height=16,width=16,steps=3)
    assert seen=={'pos','neg'} and not torch.equal(image[0],image[1])
