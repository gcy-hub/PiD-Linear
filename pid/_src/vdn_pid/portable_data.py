"""Import image/caption exports without copying images; package fixed conditions."""

import argparse
import fcntl
import hashlib
import io
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pid._src.linear_pid.data import BUCKETS, atomic_json, sha256_file


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_array(path, values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.int64))
    temporary.replace(path)


def prepare_export(package_root, output_root, source_index=None, workers=4, resume=False):
    package, out = Path(package_root).resolve(), Path(output_root).resolve()
    meta = json.loads((package / "metadata.json").read_text())
    large, small = meta["image_count"], meta["a1_a2_count"]
    rows = read_jsonl(package / "manifest.jsonl")
    selected = {r["id"] for r in read_jsonl(package / "a1_a2_samples.jsonl")}
    ids = [r["id"] for r in rows]
    if len(rows) != large or len(set(ids)) != large or len(selected) != small or not selected <= set(ids):
        raise ValueError("Export image counts, unique IDs or A1/A2 membership differ")
    source_fp = meta["signature"]
    if source_index:
        source_fp = json.loads(Path(source_index).read_text())["fingerprint"]
        settings = dict(source_fingerprint=source_fp, small=small, large=large, seed=meta["seed"])
        for size, chosen in ((small, [r for r in rows if r["id"] in selected]), (large, rows)):
            expected = hashlib.sha256(
                (source_fp + json.dumps(settings) + json.dumps(sorted(r["source_row"] for r in chosen))).encode()
            ).hexdigest()
            if meta.get("subset_fingerprints", {}).get(str(size)) != expected:
                raise ValueError("Reference index fingerprint does not reproduce exported subset fingerprints")
    identity = dict(
        manifest=sha256_file(package / "manifest.jsonl"),
        membership=sha256_file(package / "a1_a2_samples.jsonl"),
        metadata=sha256_file(package / "metadata.json"),
        source_fingerprint=source_fp,
    )
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = out / "complete.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["signature"] != signature:
                raise ValueError("Export inputs changed; use a new output directory")
            if not resume:
                raise ValueError("Prepared export exists; pass --resume")
            if all((out / p).exists() for p in previous["files"]):
                return previous

        def check(row):
            path = (package / row["image_path"]).resolve()
            if not path.is_relative_to(package):
                raise ValueError("Image path escapes the exported package")
            if not path.is_file():
                raise FileNotFoundError(path)
            if row["bucket"] not in BUCKETS or not row["caption"].strip():
                raise ValueError(f"Invalid bucket/caption for {row['id']}")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for _ in tqdm(pool.map(check, rows), total=len(rows), desc="Check exported images"):
                pass
        index = out / "index"
        index.mkdir(exist_ok=True)
        offsets, buckets, little, prompts = [], {b: [] for b in BUCKETS}, {b: [] for b in BUCKETS}, {}
        manifest = index / "manifest.jsonl"
        temporary = manifest.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            for i, row in enumerate(rows):
                record = {k: row[k] for k in ("id", "caption", "width", "height", "bucket", "source_row")}
                record.update(path=row["image_path"], split="train")
                offsets.append(stream.tell())
                stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                buckets[row["bucket"]].append(i)
                if row["id"] in selected:
                    little[row["bucket"]].append(i)
                stem, number = row["id"].split("/")
                if not stem.isdigit():
                    raise ValueError("Unexpected source caption file ID")
                number = int(number)
                if number < 0 or number >= 100000:
                    raise ValueError("Unexpected source caption row")
                prompts.setdefault(stem, {})[number] = row["caption"]
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest)
        save_array(index / "offsets.npy", offsets)
        save_array(index / "validation.npy", [])
        fingerprint = hashlib.sha256((signature + sha256_file(manifest)).encode()).hexdigest()
        base = dict(
            schema=1,
            fingerprint=fingerprint,
            source_fingerprint=source_fp,
            train_size=large,
            validation_size=0,
            seed=meta["seed"],
            dataset_root=os.path.relpath(package, index),
            export_signature=meta["signature"],
        )
        for bucket, indices in buckets.items():
            save_array(index / f"train_{bucket.replace(',', '_')}.npy", indices)
        atomic_json(index / "index.json", base)
        for size, groups in ((small, little), (large, buckets)):
            subset = out / "subsets" / str(size)
            subset.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.jsonl", "offsets.npy", "validation.npy"):
                target = subset / name
                if not target.exists():
                    target.symlink_to(os.path.relpath(index / name, subset))
            for bucket, indices in groups.items():
                save_array(subset / f"train_{bucket.replace(',', '_')}.npy", indices)
            subset_fp = hashlib.sha256(
                (fingerprint + str(size) + json.dumps(groups, sort_keys=True)).encode()
            ).hexdigest()
            atomic_json(
                subset / "index.json",
                {
                    **base,
                    "fingerprint": subset_fp,
                    "train_size": size,
                    "dataset_root": os.path.relpath(package, subset),
                },
            )
        atomic_json(out / "subsets/subsets.json", dict(small=small, large=large, source_fingerprint=fingerprint))
        # Preserve original stem/row IDs, including holes, so existing text-cache code works.
        prompt_root = out / "prompts/data_jsons"
        prompt_root.mkdir(parents=True, exist_ok=True)
        for stem, captions in prompts.items():
            records = [{} for _ in range(max(captions) + 1)]
            for number, caption in captions.items():
                records[number] = {"en_caption": caption}
            atomic_json(prompt_root / f"{stem}.json", records)
        files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != ".prepare.lock"]
        result = dict(
            signature=signature, train_size=large, a1_a2_size=small, source_fingerprint=source_fp, files=sorted(files)
        )
        atomic_json(marker, result)
        return result


def export_validation(index_root, gallery_root, output):
    index, gallery, output = Path(index_root), Path(gallery_root), Path(output)
    source = json.loads((index / "index.json").read_text())
    assets = json.loads((gallery / "assets.json").read_text())
    if assets["data_fingerprint"] != source.get("source_fingerprint", source["fingerprint"]):
        raise ValueError("Validation/source fingerprint mismatch")
    files = sorted({c["file"] for c in assets["cases"]})
    for case in assets["cases"]:
        if sha256_file(gallery / case["file"]) != case["sha256"]:
            raise ValueError(f"Corrupted validation condition: {case['file']}")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".incomplete")
    with tarfile.open(temporary, "w") as archive:
        # Imported indices have their own row layout, but the held-out conditions
        # and exported subset hashes identify the original full dataset.
        original = {**source, "fingerprint": source.get("source_fingerprint", source["fingerprint"])}
        encoded = json.dumps(original, ensure_ascii=False).encode()
        entry = tarfile.TarInfo("source_index.json")
        entry.size = len(encoded)
        archive.addfile(entry, io.BytesIO(encoded))
        archive.add(gallery / "assets.json", arcname="gallery/assets.json", recursive=False)
        for name in tqdm(files, desc="Pack fixed validation conditions"):
            archive.add(gallery / name, arcname="gallery/" + name, recursive=False)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("import")
    prepare.add_argument("--package-root", required=True, type=Path)
    prepare.add_argument("--output-root", type=Path)
    prepare.add_argument("--source-index", type=Path)
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--resume", action="store_true")
    export = commands.add_parser("export-validation")
    export.add_argument("--index-root", required=True, type=Path)
    export.add_argument("--gallery-root", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "import":
        result = prepare_export(
            args.package_root,
            args.output_root or args.package_root / "training",
            args.source_index,
            args.workers,
            args.resume,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    else:
        print(export_validation(args.index_root, args.gallery_root, args.output))


if __name__ == "__main__":
    main()
