"""Resource-bound acceptance: native weights/encoders and the integrated engine.

Run through scripts/generate_attention_bindings.py; no replacement models are used.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from attention_framework.config import ALGORITHMS, compose_config


def config(spec, family, algorithm='sliding_qdm', strategy='clear', extra=()):
    return compose_config(spec['models'][family], str(ROOT / f'configs/algorithms/{algorithm}.yaml'),
                          str(ROOT / f'configs/training/{strategy}.yaml'), spec['data'],
                          spec.get('overrides', []) + spec.get('family_overrides', {}).get(family, []) + list(extra))


def close(a, b, exact=False):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0 if exact else .02, atol=0 if exact else .02)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            close(a[key], b[key], exact)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, exact)
    elif hasattr(a, 'shape'):
        assert (a == b).all()
    else:
        assert a == b, (a, b)


def family_check(spec, family, out):
    from algorithms import build_attention
    from models import build_model
    from data import build_dataloader
    from data.prepare import prepare
    from data.encoders import NativeEncoder
    from unittest.mock import patch
    cfg = config(spec, family, extra=['runtime.gpus=1', 'training.batch_size=1',
                                     'data.caption_dropout=0', 'cache.image=false', 'cache.text=false',
                                     f'cache.root={out / "cache"}'])
    OmegaConf.save(cfg, out / 'resolved.yaml')
    # Use the whole manifest to include empty/variable text and different buckets.
    loader = build_dataloader(cfg, rank=0, world_size=1)
    count = len(loader.dataset)
    captions = [s.caption for s in loader.dataset.samples]
    assert '' in captions and len(set(map(len, captions))) > 1, 'Fixture needs empty and variable captions'
    assert len(set(loader.dataset.sizes)) > 1, 'Fixture needs multiple buckets'
    online = {}
    for index in range(count):
        batch = loader.collate_fn([loader.dataset[index]])
        online[batch.sample_ids] = batch
    del loader
    torch.cuda.empty_cache()
    prepare(cfg, devices=['cuda:0'], batch_size=1)
    for image, text in itertools.product((False, True), repeat=2):
        cfg.cache.image, cfg.cache.text = image, text
        loader = build_dataloader(cfg, rank=0, world_size=1)
        encoder = loader.collate_fn.encoder
        assert isinstance(encoder, NativeEncoder)
        assert hasattr(encoder, 'vae') == (not image)
        assert hasattr(encoder, 'text') == (not text)
        seen = set()
        for index in range(count):
            batch = loader.collate_fn([loader.dataset[index]])
            expected = online[batch.sample_ids]
            close(batch.image, expected.image)
            close(batch.text, expected.text)
            if family == 'pid':
                assert batch.image['pixels'].shape[1] == 3
                assert batch.image['pixels'].shape[-1] > batch.image['condition'].shape[-1]
            seen.add(batch.sample_ids)
        assert len(seen) == count
        del loader, encoder
        torch.cuda.empty_cache()
    batch = next(iter(online.values()))
    cfg.cache.image = cfg.cache.text = False
    for algorithm in ALGORITHMS:
        cfg = config(spec, family, algorithm, extra=['runtime.gpus=1'])
        OmegaConf.save(cfg, out / f'{algorithm}.yaml')
        torch.manual_seed(42)
        model = build_model(cfg, device=torch.device('cuda'), algorithm_factory=build_attention)
        assert {s.layer.name for s in model.sites} == set(cfg.selected_layers)
        model.requires_grad_(False)
        for role in model.role_parameters():
            role.parameter.requires_grad_(True)
        encoded_before = {k: v.clone() for k, v in batch.image.items()}
        train = model.prepare_batch(batch, generator=torch.Generator(device='cuda').manual_seed(7))
        other = model.prepare_batch(batch, generator=torch.Generator(device='cuda').manual_seed(8))
        assert not torch.equal(train.target, other.target)
        close(batch.image, encoded_before, True)
        if family == 'pid':
            assert not torch.equal(train.inputs['lq_latent'], other.inputs['lq_latent'])
        with torch.no_grad(), torch.autocast('cuda', dtype=getattr(torch, cfg.runtime.dtype),
                                            enabled=cfg.runtime.dtype != 'float32'):
            teacher = model(train, teacher=True)
        with torch.autocast('cuda', dtype=getattr(torch, cfg.runtime.dtype),
                            enabled=cfg.runtime.dtype != 'float32'):
            output = model(train, capture=tuple(cfg.feature_layers))
        assert output.prediction.shape == train.target.shape
        loss = output.prediction.float().square().mean()
        for feature in output.features.values():
            assert feature is not None
            loss = loss + feature.float().square().mean()
        loss.backward()
        assert any(r.parameter.grad is not None for r in model.role_parameters())
        assert all(torch.isfinite(r.parameter.grad).all() for r in model.role_parameters() if r.parameter.grad is not None)
        with torch.no_grad(), torch.autocast('cuda', dtype=getattr(torch, cfg.runtime.dtype),
                                            enabled=cfg.runtime.dtype != 'float32'):
            for role in model.role_parameters():
                if role.parameter.grad is not None:
                    role.parameter.add_(role.parameter.grad, alpha=-.001)
            close(model(train, teacher=True).prediction, teacher.prediction, True)
        reference = teacher.prediction.detach().cpu()
        # Reload the same native checkpoint with no sites for independent parity.
        del model, output, teacher, loss
        torch.cuda.empty_cache()
        cfg.layer_specs = []
        native = build_model(cfg, device=torch.device('cuda'), algorithm_factory=build_attention)
        with torch.no_grad(), torch.autocast('cuda', dtype=getattr(torch, cfg.runtime.dtype),
                                            enabled=cfg.runtime.dtype != 'float32'):
            close(native(train).prediction, reference)
        del native, train, other
        torch.cuda.empty_cache()
    print(f'{family}: four native algorithms, real encoder/cache parity and component absence passed')


def cycle(spec, family, out):
    from training.checkpoint import latest, read
    for strategy in ('vdn', 'clear'):
        states = []
        for label, every in (('continuous', 100), ('resumed', 1)):
            root = out / strategy / label
            cfg = config(spec, family, strategy=strategy, extra=[f'paths.output={root}',
                'runtime.gpus=4', 'training.steps=4', 'training.accumulation=2',
                'training.vdn.a1.steps=2', 'training.vdn.a2.steps=2', 'training.vdn.b.steps=2',
                f'checkpoint.every={every}', 'validation.every=2', 'validation.steps=2'])
            root.mkdir(parents=True, exist_ok=False)
            resolved = root / 'input.yaml'
            OmegaConf.save(cfg, resolved)
            resume = None
            stages = []
            while True:
                argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                        str(Path(__file__).with_name('deterministic_training.py')), '--worker-config', str(resolved)]
                if resume:
                    argv += ['--resume', str(resume)]
                subprocess.run(argv, check=True, cwd=ROOT)
                resume = latest(root)
                manifest = json.loads((resume / 'complete.json').read_text())
                stages.append(manifest['stage'])
                if manifest['pending_validation'] or manifest['done']:
                    if manifest['done']:
                        interrupted = subprocess.run([sys.executable,
                            str(ROOT / 'tests/attention_framework/interrupt_validation.py'),
                            '--config', str(resolved), '--checkpoint', str(resume)], cwd=ROOT)
                        assert interrupted.returncode != 0
                        assert (resume / 'validation/sample.png').is_file()
                        assert not (resume / 'validated.json').exists()
                    subprocess.run([sys.executable, '-m', 'training.validate', '--config', str(resolved),
                                    '--checkpoint', str(resume)], check=True, cwd=ROOT)
                    assert (resume / 'validation/sample.png').is_file()
                    assert (resume / 'validated.json').is_file()
                    if manifest['done']:
                        (resume / 'validated.json').unlink()
                        # Flatten operational config leaves into the public dotlist interface.
                        def leaves(node, prefix=''):
                            for key, value in node.items():
                                name = f'{prefix}.{key}' if prefix else key
                                if isinstance(value, dict):
                                    yield from leaves(value, name)
                                else:
                                    yield name + '=' + json.dumps(value)
                        public = {key: OmegaConf.to_container(cfg[key], resolve=True)
                                  for key in ('model', 'replacement', 'algorithm', 'training',
                                              'data', 'cache', 'runtime', 'checkpoint', 'validation', 'paths')}
                        subprocess.run([sys.executable, '-m', 'training.run',
                            '--model-config', spec['models'][family],
                            '--algorithm-config', str(ROOT / 'configs/algorithms/sliding_qdm.yaml'),
                            '--training-config', str(ROOT / f'configs/training/{strategy}.yaml'),
                            '--data-config', spec['data'], *leaves(public)], check=True, cwd=ROOT)
                        assert (resume / 'validated.json').is_file()
                if manifest['done']:
                    break
            if strategy == 'vdn':
                assert set(stages) == {'a1', 'a2', 'b'}
            states.append([read(resume, rank, 4) for rank in range(4)])
        for continuous, resumed in zip(*states):
            for key in ('weights', 'optimizer', 'scheduler', 'progress', 'rng'):
                close(continuous[key], resumed[key], True)
    print(f'{family}: real four-rank CLEAR/VDN stages, save/resume and standalone generation passed')


def delivered_cpu():
    # These checks deliberately include stand-ins; they are CPU evidence only.
    for module in ('data.check_cpu', 'algorithms.checks.cpu', 'models.check_cpu',
                   'models.check_sampling_cpu', 'training.check_cpu', 'training.check_engine_cpu',
                   'training.check_validate_cpu', 'training.check_native_cpu', 'training.check_a1_cpu'):
        subprocess.run([sys.executable, '-m', module], check=True, cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/attention_framework',
                    '-p', 'test_*.py'], check=True, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    args = parser.parse_args()
    spec = json.loads(args.resources.read_text())
    out = Path(os.environ['ACCEPTANCE_REPORT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    name = args.scenario
    family = name.removesuffix('_cycle')
    if family in ('pid', 'flux2', 'flux3'):
        cfg = config(spec, family)
        missing = [key for key, value in cfg.model.resources.items()
                   if (key != 'tokenizer' or family != 'pid') and
                   (not value or not Path(value).expanduser().exists())]
        manifest = cfg.data.manifest or cfg.paths.dataset
        if missing or not Path(manifest).exists():
            print(f'Blocked: missing native resources {missing}; dataset={manifest}', flush=True)
            raise SystemExit(77)
        if not torch.cuda.is_available():
            print('Blocked: real family checks require CUDA', flush=True)
            raise SystemExit(77)
    if name == 'config_data':
        delivered_cpu()
    elif name in ('pid', 'flux2', 'flux3'):
        if not torch.cuda.is_available():
            raise RuntimeError('Real native acceptance requires CUDA')
        torch.cuda.reset_peak_memory_stats()
        family_check(spec, name, out)
    elif name.endswith('_cycle'):
        cycle(spec, name.removesuffix('_cycle'), out)
    else:
        raise ValueError(f'No integrated runner for {name}')
    if torch.cuda.is_initialized():
        (out / 'memory.json').write_text(json.dumps(dict(allocated=torch.cuda.max_memory_allocated(),
                                                       reserved=torch.cuda.max_memory_reserved())))


if __name__ == '__main__':
    main()
