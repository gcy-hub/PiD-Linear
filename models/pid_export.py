"""Load exported PiD fixed4 weights with native Halo or Sliding attention."""
import json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file
from attention_framework.contracts import RoleParameter
from models.base import NativeModel
from pid._src.vdn_pid.config import VDNConfig
from pid._src.vdn_pid.training import build_student
from pid._src.vdn_pid.attention import execution_mode
from pid._src.common.data import sha256_file


def native_config(config, raw):
    native = VDNConfig(**{key: raw['config'][key] for key in (
        'region_size', 'local_mode', 'local_halo', 'local_shift', 'local_chunk', 'memory_mode',
        'remote_rope', 'full_attention_layers', 'lora_rank', 'lora_alpha',
        'pit_chunk_size', 'backbone')})
    if config.algorithm.name == 'qdm_halo':
        if (native.local_mode != 'halo' or native.region_size != config.algorithm.memory.region_size
                or native.local_halo != config.algorithm.memory.halo):
            raise ValueError('Halo continuation must match the exported region size and halo')
    else:
        native.local_mode = 'sliding'
        native.sliding_kernel_size = config.algorithm.local.window_size
        native.sliding_backend = 'flex'
    native.teacher_checkpoint = config.model.resources.weights
    native.activation_checkpointing = config.runtime.gradient_checkpointing
    return native


class ExportedPiD(NativeModel):
    def __init__(self, config, device):
        nn.Module.__init__(self)
        self.config, self.family = config, 'pid'
        source = Path(config.model.resources.export)
        raw = json.loads((source/'config.json').read_text())
        if raw['stage'] != 'b' or raw['metadata']['memory_mode'] != 'fixed4':
            raise ValueError('Expected PiD fixed4 B export')
        expected_base = raw.get('metadata', {}).get('base_sha256')
        if not expected_base:
            raise ValueError('Export is missing the original base fingerprint')
        base = Path(config.model.resources.weights).expanduser()
        if sha256_file(base) != expected_base:
            raise ValueError('Original base fingerprint differs from the export')
        native = native_config(config, raw)
        self.base = build_student(native, 'b', device)
        state = load_file(str(source/'model.safetensors'))
        parameters = dict(self.base.named_parameters())
        expected = {n for n in parameters if '.linear_attention.' in n or '.softmax_gate_' in n or '.A.' in n or '.B.' in n}
        if state.keys() != expected:
            raise ValueError(f'Export keys mismatch: missing={expected-state.keys()}, extra={state.keys()-expected}')
        self._roles = []
        with torch.no_grad():
            for name, weight in state.items():
                p = parameters[name]
                if p.shape != weight.shape:
                    raise ValueError(f'Export shape mismatch: {name}')
                p.copy_(weight)
                assert torch.equal(p.cpu(), weight.to(p.dtype)), name
                role = ('lora' if '.A.' in name or '.B.' in name else 'local_gate' if '.softmax_gate_' in name else 'branch') if p.requires_grad else 'frozen'
                self._roles.append(RoleParameter('base.'+name, p, role, 'patch.'+name.split('.')[1]))
        self.layers, self.sites = (), ()
        print(dict(export=str(source), loaded_tensors=len(state), source_progress=raw['progress'],
                   local=native.local_mode, region_size=native.region_size, halo=native.local_halo,
                   sliding_kernel_size=native.sliding_kernel_size if native.local_mode == 'sliding' else None), flush=True)

    def role_parameters(self):
        return self._roles

    def _forward(self, batch, *, teacher=False, capture=(), callback=None):
        if capture or callback:
            raise ValueError('Export continuation supports B prediction distillation only')
        with execution_mode(self.base, teacher=teacher):
            return super()._forward(batch, teacher=teacher)
