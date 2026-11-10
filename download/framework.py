"""Explicit downloads from model resource declarations; HF handles resumption."""
from pathlib import Path
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


def download_resources(model_config, *, components=None, workers=4):
    config = OmegaConf.load(model_config)
    resources = config.model.resources
    names = list(resources) if components is None else list(dict.fromkeys(components))
    unknown = set(names) - set(resources)
    if unknown:
        raise ValueError(f'Unknown resource components: {sorted(unknown)}')
    downloads = config.model.get('downloads', {})
    results = {}
    for name in names:
        destination = resources[name]
        if destination is None:
            raise ValueError(f'Local destination required: model.resources.{name}')
        spec = downloads.get(name)
        if spec is None:
            if Path(destination).exists():
                results[name] = str(destination)
                continue
            raise ValueError(f'Explicit download declaration required: model.downloads.{name}')
        if not spec.get('repo_id') or not spec.get('revision'):
            raise ValueError(f'Download {name} requires repo_id and revision')
        results[name] = snapshot_download(repo_id=spec.repo_id, revision=spec.revision,
            local_dir=destination, allow_patterns=spec.get('allow_patterns'), max_workers=workers,
            endpoint='https://huggingface.co')
    return results


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--components', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    download_resources(args.model_config, components=args.components, workers=args.workers)


if __name__ == '__main__':
    main()
