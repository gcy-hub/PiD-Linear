"""CUDA-event section timings, without synchronizing between network sections."""

from collections import defaultdict
import statistics

import torch


class InferenceSectionTimer:
    """Measure the MMDiT loop, PiT blocks and nested attention on the current stream.

    PiT is the sum of its two full blocks; pixel embedding, PiT condition injection,
    the final output head and image folding belong to the remaining network work.
    Call summary only after synchronizing the GPU. Nested attention times must
    not be added to their parent block times.
    """

    def __init__(self, net):
        self.net = net
        self.handles = []
        self.calls = []
        self.current = None
        self.original_loop = net._run_patch_blocks
        self.had_instance_loop = "_run_patch_blocks" in net.__dict__
        self.instance_loop = net.__dict__.get("_run_patch_blocks")

    def start(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def finish(self, label, start):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.current[label].append((start, end))

    def hook(self, module, label):
        pending = []

        def before(module, inputs):
            pending.append(self.start(label))

        def after(module, inputs, output):
            self.finish(label, pending.pop())

        self.handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

    def __enter__(self):
        def before_network(module, inputs):
            self.current = defaultdict(list)
            self.network_start = self.start("network")

        def after_network(module, inputs, output):
            self.finish("network", self.network_start)
            self.calls.append(self.current)
            self.current = None

        self.handles.extend([self.net.register_forward_pre_hook(before_network),
                             self.net.register_forward_hook(after_network)])

        def loop(*args, **kwargs):
            start = self.start("mmdit")
            output = self.original_loop(*args, **kwargs)
            self.finish("mmdit", start)
            return output

        self.net._run_patch_blocks = loop
        for index, block in enumerate(self.net.patch_blocks):
            self.hook(block, f"mmdit_block_{index}")
            self.hook(block.attn, f"mmdit_attention_{index}")
        for index, block in enumerate(self.net.pixel_blocks):
            self.hook(block, f"pit_block_{index}")
            self.hook(block.attn, f"pit_attention_{index}")
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.had_instance_loop:
            self.net._run_patch_blocks = self.instance_loop
        else:
            del self.net._run_patch_blocks

    def summary(self):
        if not self.calls:
            raise ValueError("No network calls were recorded")
        groups = defaultdict(list)
        kda = set(getattr(self.net, "kda_layers", []))
        for call in self.calls:
            values = {label: sum(a.elapsed_time(b) for a, b in events) for label, events in call.items()}
            for label, value in values.items():
                groups[label].append(value)
            pit = sum(values[f"pit_block_{i}"] for i in range(len(self.net.pixel_blocks)))
            mmdit_attention = sum(values[f"mmdit_attention_{i}"] for i in range(len(self.net.patch_blocks)))
            kda_attention = sum(values[f"mmdit_attention_{i}"] for i in kda)
            groups["pit"].append(pit)
            groups["mmdit_attention"].append(mmdit_attention)
            groups["kda_attention"].append(kda_attention)
            groups["full_attention"].append(mmdit_attention - kda_attention)
            groups["pit_attention"].append(sum(values[f"pit_attention_{i}"] for i in range(len(self.net.pixel_blocks))))
            groups["other"].append(values["network"] - values["mmdit"] - pit)
        sections = {label: {"total_ms": sum(values), "mean_ms_per_forward": statistics.mean(values),
                            "median_ms_per_forward": statistics.median(values),
                            "runs_ms": values} for label, values in groups.items()}
        total = sections["network"]["total_ms"]
        for label in ["mmdit", "pit", "other"]:
            sections[label]["fraction_of_network"] = sections[label]["total_ms"] / total
        return {"network_calls": len(self.calls), "sections": sections,
                "scope": "CUDA event elapsed times; MMDiT includes its loop and LQ injection gates; "
                         "PiT sums complete pixel blocks; nested attention timings overlap their parents"}
