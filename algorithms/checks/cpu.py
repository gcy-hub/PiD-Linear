"""Run with CUDA_VISIBLE_DEVICES='' python -m algorithms.checks.cpu."""
import copy
import torch
from omegaconf import OmegaConf
from attention_framework.contracts import AttentionInput, AttentionLayout, LayerSpec
from algorithms import build_attention
from algorithms.cross_scan_kda import scan
from algorithms.local import ClearAttention, halo_attention, sample_slices
from algorithms.qdm import joint_memory, pack_regions, unpack_regions, QuadrantMemory
from algorithms.checks.references import DenseClear, dense_quadrant_memory, StockKDAReference


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    x = torch.arange(6).reshape(1, 6, 1)
    for direction, order in enumerate(([0,1,2,3,4,5], [5,4,3,2,1,0], [0,3,1,4,2,5], [5,2,4,1,3,0])):
        assert scan(x, (2,3), direction).flatten().tolist() == order
        assert torch.equal(scan(scan(x, (2,3), direction), (2,3), direction, True), x)
    q = torch.ones(1,2,1,1, requires_grad=True)
    out, _ = joint_memory(q, torch.ones_like(q), torch.tensor([2.,4.]).reshape_as(q),
                          torch.ones(1,2,1), torch.ones(1,2,1), chunk_size=1)
    torch.testing.assert_close(out.flatten(), torch.tensor([2.,2.]))
    out.sum().backward()
    torch.testing.assert_close(q.grad.flatten(), torch.tensor([2.,2.]))
    try:
        joint_memory(q, q, q, torch.ones(1,2,1), torch.ones(1,2,1), chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid chunk accepted')
    packed, valid = pack_regions(x, (2,3), 2)
    assert valid.sum() == 6
    assert torch.equal(unpack_regions(packed, (2,3), 2), x)

    layout = AttentionLayout(((1,2,1),(1,1,0)))
    z = torch.zeros(1,1,4,1)
    out = ClearAttention(1)(z,z,torch.tensor([2.,4.,8.,16.]).reshape_as(z),layout)
    torch.testing.assert_close(out.flatten(), torch.tensor([14/3,3,5,16.]))
    for first in (True, False):
        layout = AttentionLayout(((2,4,2),(2,2,0)), first)
        q,k,v = [torch.randn(2,2,layout.length,4,requires_grad=True) for _ in range(3)]
        for shape, down in (('circle', 2), ('square', 1)):
            new = ClearAttention(2, down, window_shape=shape, kernel_size=3)
            old = DenseClear(2, down, window_shape=shape, kernel_size=3)
            a, b = new(q,k,v,layout), old(q,k,v,layout)
            torch.testing.assert_close(a,b)
            ga = torch.autograd.grad(a.square().sum(), (q,k,v,*new.parameters()))
            gb = torch.autograd.grad(b.square().sum(), (q,k,v,*old.parameters()))
            for aa,bb in zip(ga,gb): torch.testing.assert_close(aa,bb)
        # Halo reference: each image query sees its region plus halo and text.
        layout = AttentionLayout(((3,5,2),(1,2,0)), first)
        q,k,v = [torch.randn(2,layout.length,2,4,requires_grad=True) for _ in range(3)]
        for halo in (0,1):
            mask = torch.zeros(layout.length,layout.length,dtype=torch.bool)
            for (h,w),whole,image,text in sample_slices(layout):
                mask[text,whole] = True
                mask[image,text] = True
                for iy in range(h):
                    for ix in range(w):
                        for ky in range(h):
                            for kx in range(w):
                                if iy//2*2-halo <= ky < iy//2*2+2+halo and ix//2*2-halo <= kx < ix//2*2+2+halo:
                                    mask[image.start+iy*w+ix,image.start+ky*w+kx] = True
            expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
            actual = halo_attention(q,k,v,layout,2,halo)
            torch.testing.assert_close(actual,expected)
            ga = torch.autograd.grad(actual.square().sum(), (q,k,v), retain_graph=True)
            gb = torch.autograd.grad(expected.square().sum(), (q,k,v))
            for aa,bb in zip(ga,gb): torch.testing.assert_close(aa,bb,atol=2e-5,rtol=2e-5)

        cfg = OmegaConf.load('configs/default.yaml')
        cfg.algorithm.memory.region_size = 2
        for name in ('clear','sliding_qdm','qdm_halo'):
            cfg.algorithm.name = name
            spec = LayerSpec('joint.4','joint',4,9,'post_o' if 'qdm' in name else 'pre_o')
            op = build_attention(cfg,hidden_size=8,num_heads=2,layer=spec)
            hidden = torch.randn(2,layout.length,8)
            rawq, rawk = torch.randn_like(q),torch.randn_like(k)
            inp = AttentionInput(hidden,rawq,rawk,v,q,k,layout,spec)
            result = op(inp)
            assert result.pre_o.shape == hidden.shape
            roles = op.role_parameters()
            assert len({id(r.parameter) for r in roles}) == len(list(op.parameters()))
            loss = result.pre_o.square().sum()
            if result.post_o_residual is not None:
                for grid,whole,image,text in sample_slices(layout):
                    assert torch.count_nonzero(result.post_o_residual[:,text]) == 0
                loss = loss + result.post_o_residual.square().sum()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in op.parameters())
            # Changing the other packed sample cannot affect the first.
            inp2 = copy.copy(inp)
            h,w,t = layout.samples[0]; end = h*w+t
            for field in ('hidden','q_raw','k_raw','v','q_pos','k_pos'):
                value = getattr(inp,field).detach().clone(); value[:,end:] += 10
                setattr(inp2,field,value)
            changed = op(inp2)
            torch.testing.assert_close(result.pre_o[:,:end],changed.pre_o[:,:end])
            if result.post_o_residual is not None:
                torch.testing.assert_close(result.post_o_residual[:,:end],changed.post_o_residual[:,:end])
            if result.post_o_residual is not None:
                positioned = copy.copy(inp)
                positioned.q_pos = inp.q_pos + .5
                positioned.k_pos = inp.k_pos - .25
                torch.testing.assert_close(op(positioned).post_o_residual, result.post_o_residual)
                raw = copy.copy(inp)
                raw.q_raw = inp.q_raw + .5
                raw.k_raw = inp.k_raw - .25
                raw_result = op(raw)
                torch.testing.assert_close(raw_result.pre_o, result.pre_o)
                assert not torch.allclose(raw_result.post_o_residual, result.post_o_residual)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    mixed = op(inp2)
                assert torch.isfinite(mixed.pre_o).all() and torch.isfinite(mixed.post_o_residual).all()
            q,k,v = [z.detach().requires_grad_() for z in (q,k,v)]

    new = QuadrantMemory(8,2,chunk_size=2)
    x = torch.randn(6,8,requires_grad=True)
    qkv = [torch.randn(6,2,4,requires_grad=True) for _ in range(3)]
    a,b = new(x,qkv,(2,3)),dense_quadrant_memory(new,x,qkv,(2,3))
    torch.testing.assert_close(a,b)
    ga = torch.autograd.grad(a.square().sum(), (x,*qkv,*new.parameters()))
    gb = torch.autograd.grad(b.square().sum(), (x,*qkv,*new.parameters()))
    for aa,bb in zip(ga,gb): torch.testing.assert_close(aa,bb,atol=2e-5,rtol=2e-5)
    # Stock FLA construction plus an explicit stand-in tests only packed KDA glue.
    from algorithms.cross_scan_kda import CrossScanKDA
    cfg.algorithm.name = 'cross_scan_kda'
    spec = LayerSpec('joint.4','joint',4,9,'pre_o')
    op = CrossScanKDA(cfg,128,2,spec)
    assert op.kda.direction == 1
    assert op.kda.layer.A_log.dtype == op.kda.layer.dt_bias.dtype == torch.float32
    assert {r.role for r in op.role_parameters()} == {'branch'}
    class OrderedReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(128,128,bias=False)
        def forward(self, x, past_key_values=None, use_cache=False):
            assert past_key_values is None and not use_cache
            return (x.cumsum(1),)
    op.kda.layer = OrderedReference()
    reference = StockKDAReference(128, 2, spec.ordinal)
    reference.layer = copy.deepcopy(op.kda.layer)
    probe = torch.randn(2,6,128,requires_grad=True)
    for direction in range(4):
        op.kda.direction = reference.direction = direction
        a, b = op.kda(probe,(2,3)), reference(probe,(2,3))
        torch.testing.assert_close(a,b)
        ga = torch.autograd.grad(a.square().sum(), probe)[0]
        gb = torch.autograd.grad(b.square().sum(), probe)[0]
        torch.testing.assert_close(ga,gb)
    op.kda.direction = spec.ordinal % 4
    for first in (True,False):
        layout = AttentionLayout(((2,3,2),(1,2,0)),first)
        hidden = torch.randn(2,layout.length,128,requires_grad=True)
        q,k,v = [torch.randn(2,layout.length,2,64) for _ in range(3)]
        inp = AttentionInput(hidden,q,k,v,q,k,layout,spec)
        result = op(inp)
        local = op.local(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),layout).transpose(1,2).flatten(2)
        expected = local.clone()
        for grid,whole,image,text in sample_slices(layout):
            expected[:,image] += hidden[:,image].flip(1).cumsum(1).flip(1)
        torch.testing.assert_close(result.pre_o,expected)
        assert result.post_o_residual is None
        result.pre_o.sum().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        torch.testing.assert_close(op(inp).pre_o,result.pre_o)
    print('PASS: scans, solve, padding, CLEAR pooling/reference gradients, halo dense/reference gradients, QDM extraction gradients, roles, layouts, sample isolation, raw/positioned QKV, BF16 autocast, FLA construction and KDA glue')


if __name__ == '__main__':
    main()
