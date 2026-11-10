import json
import os
from unittest.mock import patch

import pytest

from scripts.train_adapt import command


def test_config_overrides_and_effective_batch(tmp_path):
    config = tmp_path / 'recipe.json'
    config.write_text(json.dumps(['--batch-size', '1', '--memory-mode', 'fixed4']))
    with patch.dict(os.environ, {'GPU_IDS': '0,1,2,3', 'NNODES': '1'}, clear=True):
        _, cmd = command(['--config', str(config), '--batch-size', '2', '--grad-accum', '4'])
    assert cmd[cmd.index('--nproc_per_node') + 1] == '4'
    assert cmd[-2:] == ['--effective-batch', '32']
    assert cmd.index('1') < cmd.index('--grad-accum')
    assert 'pid._src.vdn_pid.training' in cmd


def test_repeat_and_multinode():
    env = {'NNODES': '2', 'NPROC_PER_NODE': '4', 'NODE_RANK': '1', 'MASTER_ADDR': 'host'}
    with patch.dict(os.environ, env, clear=True):
        _, cmd = command(['--batch-size', '4', '--grad-accum', '1'])
        assert '--standalone' not in cmd
        assert cmd[-2:] == ['--effective-batch', '32']
        with pytest.raises(SystemExit):
            command(['--mode', 'repeat'])
    with patch.dict(os.environ, {'NNODES': '1'}, clear=True):
        _, repeat = command(['--mode', 'repeat', '--stage', 'all'])
        assert repeat[2:] == ['scripts.repeat_pid_vdn', '--stage', 'all']


def test_reject_non_argument_config(tmp_path):
    path = tmp_path / 'invalid.json'
    path.write_text('{}')
    with pytest.raises(SystemExit):
        command(['--config', str(path)])
