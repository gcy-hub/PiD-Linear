#!/usr/bin/env python3
"""Check an isolated worker's actual CUDA access before loading model weights."""
import json
import os
import socket


def main():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get('CPU_THREADS', '1')))
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'Worker must see exactly one GPU; visible={os.environ.get("CUDA_VISIBLE_DEVICES")}, '
                           f'count={torch.cuda.device_count()}')
    torch.cuda.set_device(0)
    with torch.inference_mode():
        q = torch.randn(1, 2, 128, 64, device='cuda', dtype=torch.bfloat16)
        out = F.scaled_dot_product_attention(q, q, q)
        mm = q.flatten(0, 2) @ q.flatten(0, 2).T
        torch.cuda.synchronize()
        if not torch.isfinite(out).all() or not torch.isfinite(mm).all():
            raise RuntimeError('CUDA preflight produced nonfinite results')
    props = torch.cuda.get_device_properties(0)
    print(json.dumps(dict(host=socket.gethostname(), visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         device=props.name, uuid=str(props.uuid),
                         total_gib=props.total_memory / 2**30, status='passed')), flush=True)


if __name__ == '__main__':
    main()
