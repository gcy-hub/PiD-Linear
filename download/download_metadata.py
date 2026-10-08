#!/usr/bin/env python3
"""Download MultiAspect-4K-1M metadata into the data_jsons/ training layout."""

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "Owen777/UltraFlux-v1"
REVISION = "ca451fff6c7d6a8f51bf682a81ff3640fdd5117b"
ARCHIVE_NAME = "MultiAspect-4K-1M.tar.gz"


def extract_metadata(archive_path, output_root):
    """Keep existing JSONs: the image downloader updates their id/image_path fields."""
    json_root = Path(output_root) / "data_jsons"
    json_root.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    # Stream once through the gzip archive; never extract arbitrary archive paths.
    with tarfile.open(archive_path, mode="r|gz") as archive:
        with tqdm(desc="Metadata JSONs", unit="file") as progress:
            for member in archive:
                name = PurePosixPath(member.name).name
                if not member.isfile() or not name.endswith(".json") or not name[:-5].isdigit():
                    continue
                destination = json_root / name
                if destination.exists():
                    skipped += 1
                else:
                    temporary = destination.with_suffix(".json.part")
                    with archive.extractfile(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    temporary.replace(destination)
                    written += 1
                progress.update()
                progress.set_postfix(written=written, skipped=skipped)
    print(f"Metadata ready in {json_root}: {written} extracted, {skipped} already present.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("raw_data/MultiAspect-4K-1M"))
    args = parser.parse_args()
    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        revision=REVISION,
        filename=ARCHIVE_NAME,
        local_dir=args.output_root / ".downloads",
        endpoint="https://huggingface.co",
    )
    extract_metadata(archive_path, args.output_root)


if __name__ == "__main__":
    main()
