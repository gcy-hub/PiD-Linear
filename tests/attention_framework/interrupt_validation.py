"""Inject a process failure after actual native PNG generation, before completion."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omegaconf import OmegaConf
import models
from training.validate import validate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
args = parser.parse_args()
native_sample = models.sample

def interrupted(*a, **kw):
    native_sample(*a, **kw)
    raise RuntimeError('Acceptance interruption after native sampling')

models.sample = interrupted
validate(OmegaConf.load(args.config), Path(args.checkpoint))
