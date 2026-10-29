"""Import the unmodified native DiT without DiffSynth's model download entrypoints.

Only package namespaces are created here; native source and relative core attention /
gradient helpers are executed directly from the user's DiffSynth checkout.
"""

import importlib
import sys
import types
from pathlib import Path


def native_module():
    root = Path(__file__).resolve().parents[2] / "DiffSynth-Studio" / "diffsynth"
    prefix = "_flux3_qdm_native"
    for suffix, directory in (("", root), (".models", root / "models"), (".core", root / "core")):
        name = prefix + suffix
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            sys.modules[name] = package
    return importlib.import_module(prefix + ".models.flux3_dit")
