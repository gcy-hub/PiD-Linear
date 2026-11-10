"""Model-independent attention operators."""
from omegaconf import DictConfig
from torch import nn
from attention_framework.contracts import LayerSpec


def build_attention(config: DictConfig, *, hidden_size: int, num_heads: int,
                    layer: LayerSpec) -> nn.Module:
    name = config.algorithm.name
    if name == 'clear':
        from .clear import CLEAR
        return CLEAR(config, layer)
    if name == 'cross_scan_kda':
        from .cross_scan_kda import CrossScanKDA
        return CrossScanKDA(config, hidden_size, num_heads, layer)
    if name == 'sliding_qdm':
        from .sliding_qdm import SlidingQDM
        return SlidingQDM(config, hidden_size, num_heads, layer)
    if name == 'qdm_halo':
        from .qdm_halo import QDMHalo
        return QDMHalo(config, hidden_size, num_heads, layer)
    raise ValueError(f'Unknown attention algorithm: {name}')
