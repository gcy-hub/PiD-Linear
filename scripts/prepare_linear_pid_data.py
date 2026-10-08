"""Validate all local images and build a resumable Linear-PiD index."""

import argparse

from pid._src.linear_pid.data import prepare_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--image-verification", choices=("full", "header"), default="full",
        help="full decodes every image; header defers full decoding to the training loader",
    )
    parser.add_argument(
        "--max-json-files", type=int, help="Optional small verification subset; omit for the full dataset"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.validation_size < 0:
        parser.error("workers must be positive and validation-size nonnegative")
    if args.max_json_files is not None and args.max_json_files < 1:
        parser.error("max-json-files must be positive")
    prepare_index(**vars(args))


if __name__ == "__main__":
    main()
