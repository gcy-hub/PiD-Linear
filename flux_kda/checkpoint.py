"""Local optimizer state I/O for fixed-world-size ZeRO-1 training."""
import torch


def move_optimizer_state(optimizer, device):
    local = getattr(optimizer, 'optim', optimizer)
    for state in local.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_optimizer(optimizer, root, step, rank):
    path = root / f'optimizer-{step}-rank-{rank}.pt'
    temporary = path.with_suffix('.partial.pt')
    torch.save(optimizer.optim.state_dict(), temporary)
    temporary.replace(path)


def load_optimizer(optimizer, root, step, rank):
    state = torch.load(root / f'optimizer-{step}-rank-{rank}.pt', map_location='cpu', weights_only=False)
    optimizer.optim.load_state_dict(state)
    # ZeRO copies outer groups into the local optimizer at the next step.
    for outer, local in zip(optimizer.param_groups, optimizer.optim.param_groups):
        outer.update({key: value for key, value in local.items() if key != 'params'})
