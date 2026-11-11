"""Create resumable 20K/50K training subsets referencing an existing data index."""

import argparse
import json

from pid._src.vdn_pid.config import VDNConfig
from pid._src.vdn_pid.data import prepare_subsets


def main():
    defaults = VDNConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-root", default=defaults.index_root)
    p.add_argument("--output-root", default=defaults.subset_root)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Preparation always resumes completed shards.")
    args = p.parse_args()
    print(
        json.dumps(prepare_subsets(args.index_root, args.output_root, workers=args.workers, seed=args.seed), indent=2)
    )


if __name__ == "__main__":
    main()
