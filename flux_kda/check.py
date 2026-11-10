"""Small CPU invariants and optional real CUDA forward/backward checks."""
import argparse
import torch
from .attention import ImageKDA, scan
from flux2_qdm.clear_strategy import objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    x = torch.arange(6).reshape(1, 6, 1)
    expected = [[0,1,2,3,4,5], [5,4,3,2,1,0], [0,3,1,4,2,5], [5,2,4,1,3,0]]
    for direction in range(4):
        result = scan(x, (2, 3), direction)
        assert result.flatten().tolist() == expected[direction]
        assert torch.equal(scan(result, (2, 3), direction, True), x)
    a = torch.ones(2, requires_grad=True)
    terms = objective(a, torch.zeros(2), torch.zeros(2), [a], [torch.zeros(2)])
    assert terms['loss'].item() == 2.
    terms['loss'].backward()
    assert torch.equal(a.grad, torch.full_like(a, 2.))
    print('CPU scan and CLEAR loss checks passed', flush=True)
    if not args.cuda:
        return
    from flux2_qdm.diffsynth_model import import_framework, LocalAttention
    import_framework()
    from diffsynth.models.flux2_dit import Flux2Attention, Flux2ParallelSelfAttention
    for direction in range(4):
        module = ImageKDA(128, 2, direction).cuda()
        z = torch.randn(2, 35, 128, device='cuda', dtype=torch.bfloat16)
        out = module(z, (5, 7))
        from torch.func import functional_call
        weights = {n: p if n in ('A_log', 'dt_bias') else p.to(z.dtype)
                   for n, p in module.layer.named_parameters()}
        expected = functional_call(module.layer, weights, (scan(z, (5, 7), direction),),
                                   dict(past_key_values=None, use_cache=False))[0]
        torch.testing.assert_close(out, scan(expected, (5, 7), direction, True), rtol=0, atol=0)
        assert torch.count_nonzero(out) > 0
        out.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        with torch.inference_mode():
            reference = module(z, (5, 7))
            module.inference_weights()
            torch.testing.assert_close(module(z, (5, 7)), reference, rtol=0, atol=0)
        print(f'CUDA KDA direction {direction}: standard-layer parity, finite gradients and BF16 inference passed', flush=True)
    for single in (False, True):
        orig = (Flux2ParallelSelfAttention(128, heads=2, dim_head=64) if single else
                Flux2Attention(128, heads=2, dim_head=64, added_kv_proj_dim=128)).cuda().bfloat16()
        layer = LocalAttention(orig, single, 0, 3, 4, 4, 'clear')
        layer.kda = ImageKDA(128, 2, 0).cuda()
        z = torch.randn(1, 38 if single else 35, 128, device='cuda', dtype=torch.bfloat16)
        kw = dict(grid=(5,7), text_length=3)
        if not single:
            kw['encoder_hidden_states'] = torch.randn(1,3,128,device='cuda',dtype=torch.bfloat16)
        output = layer(z, **kw)
        loss = sum(t.float().square().mean() for t in output) if isinstance(output, tuple) else output.float().square().mean()
        loss.backward()
        assert layer.kda.layer.o_proj.weight.grad is not None
        teacher = layer(z, teacher_mode=True, **kw)
        print(f'FLUX2 single={single}: forward, backward and teacher passed', flush=True)
    from .flux3 import Flux3KDABlock
    from flux3_qdm.attention import native, SampleLayout
    from torch.utils.checkpoint import checkpoint
    orig = native.Block(256, 2, 2).cuda().bfloat16()
    block = Flux3KDABlock(orig, 0, 3).cuda()
    block.requires_grad_(False)
    for name, param in block.named_parameters():
        if '.kda.' in '.' + name or name.endswith('train_weight'):
            param.requires_grad_(True)
    z = torch.randn(38, 256, device='cuda', dtype=torch.bfloat16)
    freqs = native.rope_embed(torch.zeros(1,38,4,device='cuda'))
    mods = (torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z))
    kw = dict(samples=(SampleLayout(0,38,0,35,35,38,(5,7)),), return_attention_feature=True)
    with torch.no_grad():
        teacher, features = block(z, freqs, [(0,38)], mods, teacher_mode=True, **kw)
        reference = orig(z, freqs, [(0,38)], mods)
        torch.testing.assert_close(teacher, reference)
        orig.q_proj.train_weight.add_(.1)
        teacher2, _ = block(z, freqs, [(0,38)], mods, teacher_mode=True, **kw)
        torch.testing.assert_close(teacher, teacher2, rtol=0, atol=0)
    student, sf = checkpoint(lambda x: block(x, freqs, [(0,38)], mods, **kw), z, use_reentrant=False)
    objective(student, teacher, torch.zeros_like(student), [sf], [features])['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters() if p.requires_grad)
    from flux2_qdm.clear_strategy import make_optimizer
    from .model import delta_state_dict, load_delta
    optimizer, scheduler = make_optimizer(p for p in block.parameters() if p.requires_grad)
    optimizer.step(); scheduler.step()
    holder = torch.nn.Module()
    holder.add_module('block', block)
    saved = {n: p.clone() for n, p in delta_state_dict(holder).items()}
    with torch.no_grad():
        block.kda.layer.o_proj.weight.add_(1)
    load_delta(holder, saved)
    assert all(torch.equal(p.cpu(), saved[n]) for n, p in holder.named_parameters() if n in saved)
    print('FLUX3 checkpointed forward/backward, feature KD, teacher isolation, Prodigy and delta restore passed', flush=True)


if __name__ == '__main__':
    main()
