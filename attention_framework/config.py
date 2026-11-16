"""OmegaConf composition and boundary validation for all framework entry points."""
import argparse
from pathlib import Path
import math

from omegaconf import DictConfig, OmegaConf

FAMILIES = ('pid', 'flux2', 'flux3')
ALGORITHMS = ('cross_scan_kda', 'sliding_qdm', 'qdm_halo', 'clear')
COMBINATIONS = tuple((m, a, s) for m in FAMILIES for a in ALGORITHMS
                     for s in ('vdn', 'clear') if (a, s) != ('clear', 'vdn'))
CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs'


def feature_tap(algorithm: str) -> str:
    return 'post_o' if algorithm in ('sliding_qdm', 'qdm_halo') else 'pre_o'


def compose_config(model_config: str, algorithm_config: str, training_config: str,
                   data_config: str, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_ROOT / 'default.yaml')
    model = OmegaConf.load(model_config)
    # Layer maps are family-specific, so replace rather than merge these two maps.
    cfg.model.layer_counts = model.model.layer_counts
    cfg.replacement.layers = model.replacement.layers
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.model.downloads, False)
    cfg = OmegaConf.merge(cfg, model, OmegaConf.load(algorithm_config),
                          OmegaConf.load(training_config), OmegaConf.load(data_config),
                          OmegaConf.from_dotlist(overrides or []))
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    for key in ('weights', 'dataset', 'output'):
        cfg.paths[key] = str(Path(cfg.paths[key]).expanduser().resolve())
    for key in ('data.manifest', 'cache.root', 'checkpoint.resume'):
        value = OmegaConf.select(cfg, key)
        if value is not None:
            OmegaConf.update(cfg, key, str(Path(value).expanduser().resolve()))
    # Resources may be HF identifiers, not filesystem paths; loaders interpret them.
    cfg._set_flag('struct', False)
    cfg.global_batch_size = cfg.training.batch_size * cfg.runtime.gpus * cfg.training.accumulation
    cfg.feature_tap = feature_tap(cfg.algorithm.name)
    cfg.selected_layers = [f'{group}.{i}' for group, selection in cfg.replacement.layers.items()
                           for i in (range(cfg.model.layer_counts[group]) if selection == 'all' else sorted(selection))]
    ordinal = 0
    cfg.layer_specs = []
    for group, count in cfg.model.layer_counts.items():
        for index in range(count):
            name = f'{group}.{index}'
            if name in cfg.selected_layers:
                cfg.layer_specs.append(dict(name=name, group=group, index=index, ordinal=ordinal,
                                           feature_tap=cfg.feature_tap))
            ordinal += 1
    cfg.feature_layers = (list(cfg.selected_layers) if cfg.replacement.feature_layers == 'all'
                          else list(cfg.replacement.feature_layers))
    if not set(cfg.feature_layers) <= set(cfg.selected_layers):
        raise ValueError('Feature layers must be selected replacement layers')
    OmegaConf.set_struct(cfg, True)
    return cfg


def validate_config(cfg: DictConfig) -> None:
    start = cfg.training.vdn.get('start_stage')
    if start not in (None, 'a1', 'a2', 'b'):
        raise ValueError('VDN start_stage must be a1, a2 or b')
    for stage in ('a1', 'a2', 'b'):
        phase = cfg.training.vdn[stage]
        if type(phase.steps) is not int or phase.steps < 1:
            raise ValueError(f'training.vdn.{stage}.steps must be a positive integer')
        if type(phase.warmup) is not int or phase.warmup < 0:
            raise ValueError(f'training.vdn.{stage}.warmup must be a nonnegative integer')
        for key in ('checkpoint_every', 'validation_every'):
            value = phase.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'training.vdn.{stage}.{key} must be a nonnegative integer or null')
    logging = cfg.get('logging')
    if logging:
        if logging.get('backend', 'none') not in ('none', 'swanlab'):
            raise ValueError('logging.backend must be none or swanlab')
        if logging.get('mode', 'offline') not in ('offline', 'online', 'local', 'disabled'):
            raise ValueError('Invalid logging.mode for SwanLab')
    wavelet = cfg.training.wavelet
    if (any(not math.isfinite(float(v)) for v in wavelet.values())
            or wavelet.weight < 0 or not 0 < wavelet.c_min <= wavelet.c_max
            or wavelet.gamma <= 0 or wavelet.alpha < 0 or wavelet.beta < 0):
        raise ValueError('Invalid SNR-HW parameters')
    if (cfg.model.family, cfg.algorithm.name, cfg.training.strategy) not in COMBINATIONS:
        raise ValueError('Unsupported model/algorithm/strategy combination (CLEAR attention requires CLEAR strategy)')
    groups = {'pid': ('patch',), 'flux2': ('double', 'single'), 'flux3': ('early_image', 'joint')}
    if set(cfg.model.layer_counts) != set(groups[cfg.model.family]):
        raise ValueError('Invalid native layer groups')
    for group, count in cfg.model.layer_counts.items():
        if type(count) is not int or count < 1:
            raise ValueError('Layer counts must be positive integers')
    for group, indices in cfg.replacement.layers.items():
        if group not in cfg.model.layer_counts:
            raise ValueError(f'Unknown layer group: {group}')
        if indices != 'all':
            if not OmegaConf.is_list(indices) or any(type(i) is not int or i < 0 or i >= cfg.model.layer_counts[group]
                                                    for i in indices) or len(set(indices)) != len(indices):
                raise ValueError(f'Invalid layer indices: {group}')
    for value in (cfg.training.batch_size, cfg.training.accumulation, cfg.runtime.gpus, cfg.training.steps,
                  cfg.algorithm.local.window_size, cfg.algorithm.memory.region_size,
                  cfg.algorithm.memory.chunk_size, cfg.algorithm['clear'].radius, cfg.algorithm['clear'].down_factor):
        if type(value) is not int or value < 1:
            raise ValueError('Batch, steps, world size and window sizes must be positive integers')
    if cfg.algorithm.local.window_size % 2 != 1 or cfg.algorithm.memory.halo < 0:
        raise ValueError('Sliding window must be odd and halo nonnegative')
    if cfg.algorithm.memory.banks != 4:
        raise ValueError('QDM uses four fixed regions')
    for value in (cfg.cache.image, cfg.cache.text):
        if type(value) is not bool:
            raise ValueError('Cache switches must be booleans')
    if (cfg.cache.image or cfg.cache.text) and not cfg.cache.root:
        raise ValueError('Enabled cache requires cache.root')
    if cfg.runtime.dtype not in ('float32', 'bfloat16'):
        raise ValueError('Training requires float32 or bfloat16; float16 requires loss scaling')


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Attention experiment configuration')
    for section, default in (('model', 'flux3'), ('algorithm', 'cross_scan_kda'),
                             ('training', 'clear'), ('data', 'default')):
        directory = {'model': 'models', 'algorithm': 'algorithms', 'training': 'training', 'data': 'data'}[section]
        parser.add_argument(f'--{section}-config', default=str(CONFIG_ROOT / directory / f'{default}.yaml'))
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('overrides', nargs='*')
    return parser


def config_from_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, DictConfig]:
    args = config_parser().parse_args(argv)
    return args, compose_config(args.model_config, args.algorithm_config, args.training_config,
                                args.data_config, args.overrides)


if __name__ == '__main__':
    _, config = config_from_args()
    print(OmegaConf.to_yaml(config, resolve=True))
