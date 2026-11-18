"""CPU checks for the two-prompt global decoder launcher."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from inference import generate_pid_8k_two_prompts as gen


@pytest.mark.parametrize('mode', ['original', 'export'])
def test_config_and_loader_use_distinct_native_branches(tmp_path, mode):
    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create({'model': {'resources': {}}, 'runtime': {}}), tmp_path / 'config.yaml')
    config = gen.config_for(tmp_path, tmp_path, mode)
    assert config.model.family == 'pid'
    assert config.model.resources.export == (str(tmp_path) if mode == 'export' else None)
    with patch('models.base.load_native') as original, patch('models.build_model') as exported, \
         patch('models.attention.fuse_lora_projections', return_value=14) as fuse:
        net, count = gen.load_decoder(config, 'cpu', mode)
    if mode == 'original':
        original.assert_called_once_with(config, 'cpu')
        exported.assert_not_called()
        fuse.assert_not_called()
        assert count == 0
    else:
        original.assert_not_called()
        assert exported.call_args.kwargs['device'] == 'cpu'
        fuse.assert_called_once_with(exported.return_value.base)
        assert count == 14


@pytest.mark.parametrize('mode', ['original', 'export'])
def test_global_sample_and_per_image_timing(tmp_path, mode):
    args = SimpleNamespace(repo=tmp_path, output=tmp_path, weights_root=tmp_path,
                           model=mode, job=0, seed=1045, steps=25, cfg=5., shift=6.)
    events = []
    clock = iter([0., 10., 20., 30., 40.])
    text = {'positive': {'embeddings': torch.ones(1), 'mask': torch.ones(1)},
            'negative': {'embeddings': torch.zeros(1), 'mask': torch.zeros(1)}}
    image = MagicMock()
    image.__enter__.return_value = SimpleNamespace(size=(8192, 8192), mode='RGB')
    def now():
        events.append('clock')
        return next(clock)
    def decoder(*args):
        events.append('load')
        return MagicMock(), 0
    def sample(*args, **kwargs):
        events.append('sample')
        assert kwargs == dict(height=8192, width=8192, seed=1045, steps=25, cfg=5., shift=6., sigma=0.)
        return torch.zeros(1, 3, 2, 2)
    with patch.object(gen, 'config_for'), patch.object(gen, 'encode_text', side_effect=lambda *a: events.append('text')), \
         patch.object(gen, 'make_latent', side_effect=lambda *a: events.append('latent')), \
         patch.object(gen, 'load_decoder', side_effect=decoder), \
         patch.object(gen.torch.cuda, 'set_device'), \
         patch.object(gen.torch.cuda, 'synchronize', side_effect=lambda *a: events.append('sync')), \
         patch.object(gen.torch, 'load', side_effect=[text, {'latent': torch.zeros(1, 16, 256, 256)}]), \
         patch('pid._src.common.runtime.sample', side_effect=sample) as sampler, \
         patch('pid._src.vdn_pid.attention.execution_mode', return_value=nullcontext()) as execution, \
         patch.object(gen.time, 'perf_counter', side_effect=now), \
         patch.object(gen.Image, 'fromarray') as png, patch.object(gen.Image, 'open', return_value=image):
        png.return_value.save.side_effect = lambda *a, **k: events.append('png')
        gen.generate_job(args)
    assert events == ['clock', 'text', 'latent', 'clock', 'load', 'sync', 'clock',
                      'sample', 'sync', 'clock', 'png', 'clock']
    sampler.assert_called_once()
    assert sampler.call_args.args[-1].shape == (1, 16, 256, 256)
    assert execution.call_count == (mode == 'export')
    png.return_value.save.assert_called_once_with(tmp_path / 'pid-8192.png', compress_level=1)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert (result['generation_seconds'], result['decode_seconds'], result['sampling_seconds']) == (40., 30., 10.)
    assert result['global_decode'] and result['model'] == mode


def test_real_encoder_and_latent_interfaces(tmp_path):
    # Exercise repository wrappers with only heavyweight model construction replaced.
    from pid._src.inference.pipeline_registry import PIPELINE_REGISTRY
    spec = PIPELINE_REGISTRY['zimage-turbo']
    encoder = MagicMock()
    encoder.encode_text.return_value = {'embeddings': torch.ones(1), 'mask': torch.ones(1)}
    pipeline = MagicMock()
    pipeline.return_value.images = torch.zeros(1, 16, 256, 256)
    with patch('data.encoders.build_encoder', return_value=encoder) as build, \
         patch('diffusers.ZImagePipeline.from_pretrained', return_value=pipeline) as pretrained, \
         patch.object(gen.torch, 'Generator'), patch.object(gen.torch.cuda, 'synchronize'):
        gen.encode_text('config', 'prompt', torch.device('cpu'), tmp_path)
        gen.make_latent(tmp_path, 'prompt', 1045, torch.device('cuda', 0), tmp_path)
    build.assert_called_once_with('config', device=torch.device('cpu'), image=False, text=True)
    assert encoder.encode_text.call_args_list[0].args == (['prompt'],)
    pretrained.assert_called_once_with(tmp_path / 'Z-Image-Turbo', torch_dtype=torch.bfloat16, local_files_only=True)
    call = pipeline.call_args.kwargs
    assert (call['height'], call['width'], call['output_type']) == (2048, 2048, 'latent')
    assert call['num_inference_steps'] == spec.default_num_inference_steps
    assert torch.load(tmp_path / 'latent.pt', weights_only=True)['latent'].shape == (1, 16, 256, 256)


@pytest.mark.parametrize('mode', ['original', 'export'])
@pytest.mark.parametrize('gpus', ['0,0', '0,1'])
def test_launcher_scheduling_and_default_outputs(tmp_path, monkeypatch, mode, gpus):
    monkeypatch.chdir(tmp_path)
    argv = ['generate', '--model', mode, '--gpus', gpus]
    if mode == 'export':
        argv += ['--repo', str(tmp_path)]
    with patch.object(gen.sys, 'argv', argv), patch.object(gen, 'run_job') as job, \
         patch.object(gen, 'ThreadPoolExecutor') as pool:
        gen.main()
    assert pool.call_count == (gpus != '0,0')
    if gpus == '0,0':
        assert [c.args[3] for c in job.call_args_list] == [0, 1]
    else:
        assert pool.return_value.__enter__.return_value.submit.call_count == 2
    result = json.loads((tmp_path / f'outputs/pid-{mode}-8k-two-prompts/run.json').read_text())
    assert result['model'] == mode


def test_worker_command_and_failure_propagation(tmp_path):
    with patch.object(gen.subprocess, 'run', side_effect=RuntimeError('failed')) as run:
        with pytest.raises(RuntimeError, match='failed'):
            gen.run_job(None, tmp_path, tmp_path, 0, '2', 1, 25, 5., 6., 'original')
    command = run.call_args.args[0]
    assert Path(command[1]).is_absolute()
    assert command[command.index('--model') + 1] == 'original' and '--repo' not in command
    assert run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2'
    assert run.call_args.kwargs['check'] is True


def test_export_requires_repo():
    with patch.object(gen.sys, 'argv', ['generate']), pytest.raises(SystemExit) as error:
        gen.main()
    assert error.value.code == 2
