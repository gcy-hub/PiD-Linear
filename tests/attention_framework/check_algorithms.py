"""CUDA operator shape, finite-gradient and packed-sample isolation checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from algorithms import build_attention
from attention_framework.config import ALGORITHMS, CONFIG_ROOT, compose_config
from attention_framework.contracts import AttentionInput, AttentionLayout, LayerSpec


def check():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for algorithm acceptance')
    for algorithm in ALGORITHMS:
        cfg = compose_config(str(CONFIG_ROOT / 'models/flux3.yaml'),
                             str(CONFIG_ROOT / f'algorithms/{algorithm}.yaml'),
                             str(CONFIG_ROOT / 'training/clear.yaml'),
                             str(CONFIG_ROOT / 'data/default.yaml'),
                             ['algorithm.local.window_size=3', 'algorithm.memory.region_size=2',
                              'algorithm.memory.halo=1', 'algorithm.clear.radius=2'])
        for text_first in (True, False):
            layout = AttentionLayout(((2, 3, 2), (2, 2, 0)), text_first)
            layer = LayerSpec('joint.4', 'joint', 4, 9, cfg.feature_tap)
            torch.manual_seed(42)
            op = build_attention(cfg, hidden_size=128, num_heads=2, layer=layer).cuda()
            tensors = [torch.randn(1, layout.length, 128, device='cuda', requires_grad=True)]
            tensors += [torch.randn(1, layout.length, 2, 64, device='cuda', requires_grad=True) for _ in range(5)]
            inputs = AttentionInput(*tensors, layout, layer)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = op(inputs)
            assert result.pre_o.shape == (1, layout.length, 128)
            assert torch.isfinite(result.pre_o).all()
            loss = result.pre_o.float().square().mean()
            if result.post_o_residual is not None:
                assert result.post_o_residual.shape == result.pre_o.shape
                assert torch.isfinite(result.post_o_residual).all()
                loss = loss + result.post_o_residual.float().square().mean()
            loss.backward()
            if algorithm != 'cross_scan_kda':
                import copy
                reference_op = copy.deepcopy(op).cpu()
                cpu_tensors = [t.detach().cpu().requires_grad_() for t in tensors]
                reference = reference_op(AttentionInput(*cpu_tensors, layout, layer))
                torch.testing.assert_close(result.pre_o.float().cpu(), reference.pre_o.float(), rtol=.04, atol=.04)
                reference_loss = reference.pre_o.float().square().mean()
                if reference.post_o_residual is not None:
                    torch.testing.assert_close(result.post_o_residual.float().cpu(), reference.post_o_residual.float(), rtol=.04, atol=.04)
                    reference_loss += reference.post_o_residual.float().square().mean()
                reference_loss.backward()
                for cuda_tensor, cpu_tensor in zip(tensors, cpu_tensors):
                    assert (cuda_tensor.grad is None) == (cpu_tensor.grad is None)
                    if cuda_tensor.grad is not None:
                        torch.testing.assert_close(cuda_tensor.grad.float().cpu(), cpu_tensor.grad.float(), rtol=.08, atol=.01)
            if algorithm == 'cross_scan_kda':
                from flux_kda.attention import ImageKDA as OriginalKDA
                from flux2.flux2_qdm.vendor.clear_attention import ClearAttention as OriginalLocal
                from flux2.flux2_qdm.vendor.clear_attention import AttentionLayout as OriginalLayout
                from algorithms.local import sample_slices
                original = OriginalKDA(128, 2, layer.ordinal).cuda()
                original.load_state_dict(op.kda.state_dict())
                reference_hidden = tensors[0].detach().clone().requires_grad_()
                original_local = OriginalLocal(window_shape='square', kernel_size=3)
                local = original_local(tensors[4].detach().cpu().transpose(1, 2),
                    tensors[5].detach().cpu().transpose(1, 2), tensors[3].detach().cpu().transpose(1, 2),
                    OriginalLayout(layout.samples, text_first)).transpose(1, 2).flatten(2).cuda()
                parts = []
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for grid, whole, image, text in sample_slices(layout):
                        remote = original(reference_hidden[:, image], grid)
                        zeros = remote.new_zeros(1, text.stop-text.start, 128)
                        parts.extend((zeros, remote) if text_first else (remote, zeros))
                    expected = local + torch.cat(parts, 1)
                torch.testing.assert_close(result.pre_o.float(), expected.float(), rtol=.04, atol=.04)
                expected.float().square().mean().backward()
                torch.testing.assert_close(tensors[0].grad.float(), reference_hidden.grad.float(), rtol=.04, atol=.01)
            assert any(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)
            # Perturb sample two only; sample one's output must remain identical.
            changed = [tensor.detach().clone() for tensor in tensors]
            for tensor in changed:
                tensor[:, 8:] += 10
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                isolated = op(AttentionInput(*changed, layout, layer))
            torch.testing.assert_close(result.pre_o[:, :8], isolated.pre_o[:, :8])
            if result.post_o_residual is not None:
                torch.testing.assert_close(result.post_o_residual[:, :8], isolated.post_o_residual[:, :8])
            print(f'{algorithm} text_first={text_first}: shape/gradient/sample isolation passed', flush=True)


if __name__ == '__main__':
    check()
