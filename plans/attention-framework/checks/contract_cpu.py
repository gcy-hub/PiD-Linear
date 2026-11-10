"""Bounded CPU checks for shared config and existing numerical references."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf.errors import ConfigKeyError
from setuptools import find_namespace_packages

from attention_framework.config import ALGORITHMS, COMBINATIONS, CONFIG_ROOT, compose_config
from attention_framework.contracts import AttentionLayout, AttentionResult, LayerSpec


def config(m='flux3', a='cross_scan_kda', s='clear', overrides=None):
    return compose_config(str(CONFIG_ROOT / f'models/{m}.yaml'),
                          str(CONFIG_ROOT / f'algorithms/{a}.yaml'),
                          str(CONFIG_ROOT / f'training/{s}.yaml'),
                          str(CONFIG_ROOT / 'data/default.yaml'), overrides)


def main():
    assert len(COMBINATIONS) == 21
    assert len({(m, a) for m, a, _ in COMBINATIONS}) == 12
    for m, a, s in COMBINATIONS:
        cfg = config(m, a, s)
        assert cfg.model.family == m and cfg.algorithm.name == a and cfg.training.strategy == s
        assert Path(cfg.paths.output).is_absolute()
        assert cfg.feature_tap == ('post_o' if 'qdm' in a else 'pre_o')
    for overrides in (['training.typo=1'], ['replacement.layers.joint=[28]'],
                      ['algorithm.local.window_size=4'], ['cache.image=true'],
                      ['cache.text=1'], ['training.batch_size=0'], ['runtime.dtype=float16'],
                      ['replacement.feature_layers=[joint.40]']):
        try:
            config(overrides=overrides)
        except (ValueError, ConfigKeyError):
            pass
        else:
            raise AssertionError(overrides)
    for m in ('pid', 'flux2', 'flux3'):
        try:
            config(m, 'clear', 'vdn')
        except ValueError:
            pass
        else:
            raise AssertionError('CLEAR+VDN accepted')
    downloaded = config(overrides=['model.downloads.vae.repo_id=example/model',
                                   'model.downloads.vae.revision=fixed'])
    assert downloaded.model.downloads.vae.repo_id == 'example/model'
    cfg = config(overrides=['training.batch_size=4', 'training.accumulation=2', 'runtime.gpus=3',
                            'replacement.layers.early_image=[]', 'replacement.layers.joint=[4,8]'])
    assert cfg.global_batch_size == 24 and list(cfg.selected_layers) == ['joint.4', 'joint.8']
    assert LayerSpec('joint.4', 'joint', 4, 9, 'pre_o').ordinal % 4 == 1
    assert AttentionLayout(((2, 3, 2), (1, 2, 1)), False).length == 11
    assert AttentionResult(torch.ones(1, 2, 3)).post_o_residual is None
    packages = find_namespace_packages(str(ROOT), include=['attention_framework*', 'models*', 'algorithms*',
                                                           'data*', 'training*', 'configs*'])
    assert all(p in packages for p in ('attention_framework', 'models', 'algorithms', 'data', 'training', 'configs'))

    # Execute the existing scan function without importing its CUDA-only FLA dependency.
    source = ast.parse((ROOT / 'flux_kda/attention.py').read_text())
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'scan'],
                            type_ignores=[]), 'flux_kda/attention.py', 'exec'), namespace)
    scan = namespace['scan']
    x = torch.arange(6).reshape(1, 6, 1)
    scans = [scan(x, (2, 3), d).flatten().tolist() for d in range(4)]
    assert scans == [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [0, 3, 1, 4, 2, 5], [5, 2, 4, 1, 3, 0]]
    for d in range(4):
        assert torch.equal(scan(scan(x, (2, 3), d), (2, 3), d, inverse=True), x)
    from flux2.flux2_qdm.memory import joint_memory, pack_regions, unpack_regions
    from flux2.flux2_qdm.vendor.clear_attention import AttentionLayout as NativeLayout, ClearAttention, clear_mask
    q = torch.ones(1, 2, 1, 1, requires_grad=True)
    memory, _ = joint_memory(q, torch.ones_like(q), torch.tensor([2., 4.]).reshape_as(q),
                             torch.ones(1, 2, 1), torch.ones(1, 2, 1))
    torch.testing.assert_close(memory, torch.full_like(memory, 2.))
    memory.sum().backward()
    torch.testing.assert_close(q.grad, torch.full_like(q, 2.))
    packed, valid = pack_regions(x.float(), (2, 3), 2)
    assert valid.sum().item() == 6
    torch.testing.assert_close(unpack_regions(packed, (2, 3), 2), x.float())
    layout = NativeLayout(((1, 2, 1), (1, 1, 0)), text_first=True)
    mask = clear_mask(layout, 1, 'cpu')
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False],
                             [True, False, True, False], [False, False, False, True]]
    z = torch.zeros(1, 1, 4, 1)
    out = ClearAttention(window_size=1)(z, z, torch.tensor([2., 4., 8., 16.]).reshape_as(z), layout)
    torch.testing.assert_close(out.flatten(), torch.tensor([14/3, 3., 5., 16.]))
    print(json.dumps({'valid_configs': 21, 'model_algorithm_pairs': 12, 'global_batch': 24,
                      'scan_reference': scans, 'qdm_reference': memory.detach().flatten().tolist(),
                      'clear_reference': out.flatten().tolist(), 'region_valid_tokens': 6}))


if __name__ == '__main__':
    main()
