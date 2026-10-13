"""Linear-cost autograd for VDN's unchanged two-direction state recurrence.

The reference Python loop creates a select-backward for every transition row.
Each such node allocates a gradient for the complete region bank. This custom
backward writes each row once, keeping both forward and backward linear in the
number of regions. Joint factorization remains the original VDN implementation.
"""

import torch
from torch.autograd.function import once_differentiable


class _BidirectionalScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transitions, injections, start):
        prefix = torch.empty_like(injections)
        suffix = torch.empty_like(injections)
        state = start
        for i in range(len(transitions)):
            torch.baddbmm(injections[i], state, transitions[i], out=prefix[i])
            state = prefix[i]
        state = start
        for i in range(len(transitions) - 1, -1, -1):
            torch.baddbmm(injections[i], state, transitions[i], out=suffix[i])
            state = suffix[i]
        ctx.save_for_backward(transitions, start, prefix, suffix)
        return prefix, suffix

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_prefix, grad_suffix):
        transitions, start, prefix, suffix = ctx.saved_tensors
        grad_transitions = torch.empty_like(transitions)
        grad_injections = torch.empty_like(prefix)
        carry = torch.zeros_like(start)
        with torch.autocast(device_type=start.device.type, enabled=False):
            for i in range(len(transitions) - 1, -1, -1):
                current = carry + grad_prefix[i]
                previous = start if i == 0 else prefix[i - 1]
                grad_transitions[i].copy_(previous.transpose(-1, -2) @ current)
                grad_injections[i].copy_(current)
                carry = current @ transitions[i].transpose(-1, -2)
            grad_start = carry
            carry = torch.zeros_like(start)
            for i in range(len(transitions)):
                current = carry + grad_suffix[i]
                previous = start if i == len(transitions) - 1 else suffix[i + 1]
                grad_transitions[i].add_(previous.transpose(-1, -2) @ current)
                grad_injections[i].add_(current)
                carry = current @ transitions[i].transpose(-1, -2)
            grad_start = grad_start + carry
        return grad_transitions, grad_injections, grad_start


def scan_bidirectional(backend, alpha, A, B, text_state=None):
    with torch.autocast(device_type=A.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, A, B)
        start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
        return _BidirectionalScan.apply(transitions, injections, start)
