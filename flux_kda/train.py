"""Train spatial KDA with CLEAR's direct projections, three losses and Prodigy."""
import argparse
import importlib
import json
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from flux2_qdm.clear_strategy import make_optimizer, objective, compute_context
from .checkpoint import save_optimizer, load_optimizer, move_optimizer_state
from .optimizer import MasterProdigy
from .model import build_model, teacher_mode, delta_state_dict, load_delta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('flux2', 'flux3'), default='flux3')
    p.add_argument('--config', help='JSON configuration; command-line options override it')
    for key in ('weights', 'subset-root', 'cache-root', 'output'):
        p.add_argument('--' + key)
    p.add_argument('--data-mode', choices=('online', 'cached'), default='online')
    p.add_argument('--resources', default=str(Path.home() / 'huggingface_ckpts'))
    p.add_argument('--max-resolution', type=int, choices=(3072,3840,4096), default=4096)
    p.add_argument('--shard-optimizer', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--activation-cpu-offload', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--optimizer-cpu-offload', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--allow-batch-change', action='store_true')
    p.add_argument('--distributed-timeout-minutes', type=int, default=60)
    p.add_argument('--subset-size', type=int, default=0, help='Optional legacy cache selection; 0 detects the dataset size')
    p.add_argument('--image-resolution', type=int, choices=(1024,2048), default=2048,
                   help='FLUX3 indexed-cache bucket configuration; CLEAR CSV cache reads its own resolution')
    p.add_argument('--allow-dataset-change', action='store_true', help='Resume model and optimizer on a new dataset')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--kernel', type=int, default=11)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume', help='Path to a CLEAR KDA checkpoint')
    p.add_argument('--text-logit-bias', help='FLUX3 per-head calibration JSON for the mixing ablation')
    p.add_argument('--no-checkpointing', action='store_true')
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text())
        known = {a.dest for a in p._actions}
        unknown = set(config) - known - {'gpus', 'validation'}
        if unknown:
            p.error(f'Unknown configuration keys: {sorted(unknown)}')
        p.set_defaults(**{k:v for k,v in config.items() if k in known})
    args = p.parse_args(argv)
    for key in ('weights', 'subset_root', 'output'):
        if not getattr(args,key):
            p.error(f'{key} is required in the config or command line')
    if args.data_mode == 'cached' and not args.cache_root:
        p.error('cached mode requires cache_root')
    if args.data_mode == 'online' and args.family != 'flux3':
        p.error('Online encoding currently supports FLUX3; use cached mode for FLUX2')
    if args.subset_size < 0 or args.distributed_timeout_minutes < 1:
        p.error('subset_size must be nonnegative and timeout positive')
    if min(args.steps, args.batch_size, args.grad_accum, args.save_every, args.kernel) < 1 or args.kernel % 2 != 1:
        p.error('Counts must be positive and the sliding kernel odd')
    if args.allow_dataset_change and not args.resume:
        p.error('--allow-dataset-change requires --resume')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.family == 'flux2':
        os.environ['FLUX2_WEIGHTS_ROOT'] = args.weights
        os.environ['FLUX2_BACKEND'] = 'diffsynth'
    rank, world, local_rank = (int(os.environ.get(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=args.distributed_timeout_minutes))
    torch.manual_seed(args.seed)
    data = importlib.import_module(f'{args.family}_qdm.data')
    runtime = importlib.import_module(f'{args.family}_qdm.runtime')
    if args.data_mode == 'cached' and args.family == 'flux3' and not args.subset_size:
        subset = Path(args.subset_root)
        if (subset / 'metadata.csv').exists():
            import csv
            with (subset / 'metadata.csv').open() as stream:
                args.subset_size = sum(1 for _ in csv.DictReader(stream))
        else:
            args.subset_size = json.loads((subset / 'index.json').read_text())['train_size']
    encoder = None
    if args.data_mode == 'online':
        from .online import OnlineDataset, OnlineEncoder
        dataset = OnlineDataset(args.subset_root, args.max_resolution)
        encoder = OnlineEncoder(args.resources, device)
        if rank == 0:
            print(f'Dataset: {len(dataset)} accepted, {dataset.rejected} below minimum; global batch {args.batch_size * world * args.grad_accum}', flush=True)
    elif args.family == 'flux3' and (Path(args.subset_root) / 'metadata.csv').exists():
        from .data import ClearFlux3Dataset
        dataset = ClearFlux3Dataset(args.subset_root, args.cache_root, args.subset_size)
    elif args.family == 'flux3':
        dataset = data.CachedDataset(args.subset_root, args.cache_root, args.subset_size,
                                     image_buckets=data.BUCKETS_1K if args.image_resolution == 1024 else data.BUCKETS_2K)
    else:
        dataset = data.CachedDataset(args.subset_root, args.cache_root, 0)
    sampler = runtime.GlobalSampler(dataset.bucket_indices, args.batch_size, world, args.grad_accum, args.seed)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'latest.pt'
    if destination.exists() and not args.resume:
        raise FileExistsError('Output has a checkpoint; pass --resume or use a new directory')
    net = build_model(args.family, args.weights, device, args.kernel)
    if args.family == 'flux3':
        for parameter in net.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)
    sharded = args.shard_optimizer and world > 1
    optimizer, scheduler = make_optimizer((p for p in net.parameters() if p.requires_grad), shard=sharded,
                                          optimizer_class=MasterProdigy if args.family == 'flux3' else None)
    identity = dict(family=args.family, weights=str(Path(args.weights).resolve()), kernel=args.kernel,
                    seed=args.seed, batch_size=args.batch_size, grad_accum=args.grad_accum, world=world,
                    data=dataset.fingerprint, architecture='sliding-stock-fla-kda-v2', policy='clear')
    if args.family == 'flux3':
        identity['training_dtype'] = 'bfloat16'
        identity['optimizer_precision'] = 'float32_master'
    if args.text_logit_bias:
        if args.family != 'flux3':
            raise ValueError('Text-logit calibration is implemented for FLUX3')
        from .mixing import apply_bias, read_bias
        values = read_bias(args.text_logit_bias)
        apply_bias(net, values)
        identity['text_logit_bias'] = values
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        saved_identity = dict(state['identity'])
        if args.family == 'flux3' and 'training_dtype' not in saved_identity:
            saved_identity['training_dtype'] = 'bfloat16'
        if args.family == 'flux3':
            saved_identity.setdefault('optimizer_precision', 'float32_master')
        if args.text_logit_bias:
            saved_identity.setdefault('text_logit_bias', identity['text_logit_bias'])
        if args.allow_dataset_change:
            saved_identity['data'] = identity['data']
        if args.allow_batch_change:
            for key in ('world','batch_size','grad_accum'):
                saved_identity[key] = identity[key]
        if saved_identity != identity:
            raise ValueError('Resume model, data, or global batch configuration differs')
        load_delta(net, state['delta'])
        if bool(state.get('optimizer_shards')) != sharded:
            raise ValueError('Resume requires the same optimizer sharding setting')
        if sharded:
            if state['optimizer_shards'] != world:
                raise ValueError('Optimizer shards require the original GPU count')
            load_optimizer(optimizer, Path(args.resume).parent, state['step'], rank)
        else:
            optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start = state['step']
        del state
    if args.family == 'flux3':
        if rank == 0:
            trainable = [p for p in net.parameters() if p.requires_grad]
            assert all(p.dtype == torch.bfloat16 for p in trainable)
            layout = 'sharded' if sharded else 'replicated'
            print(f'FLUX3 BF16 trainable parameters: {sum(p.numel() for p in trainable)}; {layout} FP32 master parameters and optimizer states', flush=True)
    if args.optimizer_cpu_offload:
        move_optimizer_state(optimizer, 'cpu')
    model = DistributedDataParallel(net, device_ids=[local_rank], broadcast_buffers=False,
                                    gradient_as_bucket_view=True) if world > 1 else net
    saved_step = start
    model.train()
    for step in range(start, args.steps):
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(4, device=device)
        for micro in range(args.grad_accum):
            samples = [dataset[i] for i in sampler.rank_batch(step, rank, micro)]
            if encoder is not None:
                samples = encoder(samples)
            packed, target = runtime.training_inputs(samples, args.seed, 'b', step, device, caption_dropout=0.)
            with torch.no_grad(), teacher_mode(net), compute_context():
                teacher, teacher_features = net(packed, collect_features=True)
            activation_context = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.activation_cpu_offload else nullcontext()
            with activation_context, compute_context():
                student, student_features = model(packed, collect_features=True, use_gradient_checkpointing=args.gradient_checkpointing and not args.no_checkpointing)
                terms = objective(student, teacher, target, student_features, teacher_features)
            finite = torch.isfinite(terms['loss']).to(torch.int)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite CLEAR loss at step {step}')
            (terms['loss'] / args.grad_accum).backward()
            metrics += torch.stack([terms[k].detach() for k in ('loss', 'loss_fm', 'loss_distill', 'loss_attn')]) / args.grad_accum
            del student, student_features, teacher, teacher_features, terms, packed, target
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], float('inf'), error_if_nonfinite=True)
        if args.optimizer_cpu_offload:
            move_optimizer_state(optimizer, device)
        optimizer.step()
        if args.optimizer_cpu_offload:
            move_optimizer_state(optimizer, 'cpu')
        scheduler.step()
        if world > 1:
            dist.all_reduce(metrics)
            metrics /= world
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        save = (step + 1) % args.save_every == 0 or step + 1 == args.steps
        if save and sharded:
            optimizer.zero_grad(set_to_none=True)
            save_optimizer(optimizer, root, step + 1, rank)
            dist.barrier()
        if rank == 0:
            record = dict(step=step + 1, step_seconds=step_seconds, peak_allocated_gib=peak_gib,
                          **dict(zip(('loss', 'fm', 'prediction_kd', 'attention_kd'), metrics.tolist())))
            print(json.dumps(record), flush=True)
            with (root / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if save:
                temporary = root / 'latest.partial.pt'
                torch.save(dict(step=step + 1, identity=identity, config=vars(args), delta=delta_state_dict(net),
                                optimizer=optimizer.state_dict() if not sharded else None,
                                optimizer_shards=world if sharded else 0, scheduler=scheduler.state_dict()), temporary)
                temporary.replace(destination)
        if world > 1:
            dist.barrier()
            if save and saved_step and sharded:
                (root / f'optimizer-{saved_step}-rank-{rank}.pt').unlink(missing_ok=True)
        if save:
            saved_step = step + 1
            if rank == 0:
                print(f'Checkpoint saved at step {saved_step}', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
