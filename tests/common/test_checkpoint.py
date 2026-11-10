import json
import random

import numpy as np
import torch

from pid._src.common.checkpoint import capture_rng, complete_checkpoints, cpu_tree, restore_rng
from pid._src.common.training import broadcast_object, consensus


def test_checkpoint_rng_roundtrip_and_complete_publication(tmp_path):
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2])
    value = cpu_tree({'weights': [torch.ones(2, requires_grad=True)]})
    assert not value['weights'][0].requires_grad
    for name, size in [('step_000001', 3), ('step_000002', 4), ('step_000003.incomplete', 3)]:
        folder = tmp_path / 'checkpoints' / name
        folder.mkdir(parents=True)
        (folder / 'state.pt').write_bytes(b'abc')
        (folder / 'complete.json').write_text(json.dumps({'bytes': size}))
    assert [p.name for p in complete_checkpoints(tmp_path)] == ['step_000001']


def test_single_rank_collectives():
    assert broadcast_object({'step': 4}, 0) == {'step': 4}
    assert consensus(True, 'cpu')
    assert not consensus(False, 'cpu')
