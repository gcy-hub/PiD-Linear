"""Prompt-only PixelDiT sampling, with optional matched original/student pairs."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from pid._src.linear_pid.data import atomic_json, sha256_file
from pid._src.linear_pid.evaluation import save_views, tensor_image
from pid._src.linear_pid.text_encoder import NEGATIVE_PROMPT
from .conditioning import PixelDiTTextEncoder
from pid._src.linear_pid.training import init_distributed
from pid._src.vdn_pid.attention import execution_mode
from pid._src.vdn_pid.checkpoint import load_delta, load_state, verify_base
from pid._src.vdn_pid.config import VDNConfig
from pid._src.vdn_pid.training import build_student
from .runtime import load_net, sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompts-file')
    parser.add_argument('--checkpoint', help='Complete QDM checkpoint directory; omit for original PixelDiT')
    parser.add_argument('--weights-root', default='weights')
    parser.add_argument('--teacher-checkpoint', default='')
    parser.add_argument('--output-dir', default='outputs/pixeldit-t2i-qdm-halo2/prompts')
    parser.add_argument('--compare-original', action='store_true')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=2.75)
    parser.add_argument('--shift', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.height, args.width, args.steps, args.threads) < 1 or args.height % 16 or args.width % 16:
        parser.error('Positive sizes divisible by 16, positive steps and threads are required')
    torch.set_num_threads(args.threads)
    rank, world, device = init_distributed()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    texts = [args.prompt] if args.prompt else [line.strip() for line in Path(args.prompts_file).read_text().splitlines() if line.strip()]
    local = list(enumerate(texts))[rank::world]
    try:
        if not local:
            return
        payload = load_state(args.checkpoint) if args.checkpoint else None
        if payload:
            config = VDNConfig(**payload['config'])
            if config.backbone != 'pixeldit_t2i':
                raise ValueError('Expected a PixelDiT-T2I QDM checkpoint')
            config.weights_root = args.weights_root
            config.teacher_checkpoint = args.teacher_checkpoint or str(Path(args.weights_root) / payload['metadata']['base_relative_path'])
            verify_base(config, payload)
        else:
            config = VDNConfig(backbone='pixeldit_t2i', weights_root=args.weights_root, teacher_checkpoint=args.teacher_checkpoint)
        # Encode once, then destroy the encoder before loading the image backbone.
        # Only its small condition tensors remain on GPU; no model CPU offload.
        encoder = PixelDiTTextEncoder(config.weights_root, device)
        conditions = []
        null_embs, null_mask = encoder.encode_raw([NEGATIVE_PROMPT])
        for ordinal, caption in local:
            emb, mask = encoder.encode_text([caption])
            conditions.append((ordinal, caption, emb, mask))
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        if payload:
            net = build_student(config, payload['stage'], device)
            load_delta(net, payload['delta'])
            identity = sha256_file(Path(args.checkpoint) / 'state.pt' if Path(args.checkpoint).is_dir() else args.checkpoint)
        else:
            net = load_net(config.teacher_path)
            net = net.to(device=device, dtype=torch.bfloat16)
            identity = sha256_file(config.teacher_path)
        del payload
        net.eval().requires_grad_(False)
        for ordinal, caption, emb, mask in tqdm(conditions, desc=f'PixelDiT prompts rank{rank}'):
            prefix = f'prompt_{ordinal:03d}_{args.height}x{args.width}'
            settings = dict(caption=caption, model_sha256=identity, seed=args.seed+ordinal,
                            height=args.height, width=args.width, steps=args.steps, cfg=args.cfg, shift=args.shift,
                            compare_original=args.compare_original, conditioning='prompt-only')
            marker = output / (prefix+'.json')
            if marker.exists():
                previous = json.loads(marker.read_text())
                if previous['settings'] != settings:
                    raise ValueError(f'Output identity changed: {marker}; choose another output directory')
                if all((output / name).exists() for name in previous['images']):
                    continue
            images, measurements = [], {}
            modes = [True, False] if args.checkpoint and args.compare_original else [not bool(args.checkpoint)]
            with torch.no_grad():
                for teacher in modes:
                    label = 'original' if teacher else 'qdm'
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    with execution_mode(net, teacher=teacher, inference=True):
                        image = sample(net, emb, mask, null_embs, null_mask, height=args.height, width=args.width,
                                       seed=args.seed+ordinal, steps=args.steps, cfg=args.cfg, shift=args.shift)
                    torch.cuda.synchronize()
                    measurements[label] = dict(seconds=time.perf_counter()-started,
                                               peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30)
                    name = prefix+'_'+label+'.png'
                    save_views(tensor_image(image), output/name)
                    images.append(name)
                    del image
            if len(images) == 2:
                with Image.open(output/images[0]) as left, Image.open(output/images[1]) as right:
                    pair = Image.new('RGB', (left.width+right.width, left.height))
                    pair.paste(left, (0, 0))
                    pair.paste(right, (left.width, 0))
                    pair.save(output/(prefix+'_comparison.png'))
            atomic_json(marker, dict(settings=settings, measurements=measurements, images=images))
            print(json.dumps(dict(prompt=ordinal, **measurements)), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
