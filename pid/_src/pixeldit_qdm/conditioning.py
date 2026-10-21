"""Reuse caption caches; native PixelDiT CFG null text has no CHI prefix."""
import argparse
import os
from pathlib import Path

import torch
from pid._src.linear_pid.text_cache import TextCacheReader
from pid._src.linear_pid.text_encoder import GemmaTextEncoder, NEGATIVE_PROMPT


class PixelDiTTextEncoder(GemmaTextEncoder):
    @torch.no_grad()
    def encode_raw(self, captions):
        tokens = self.tokenizer(captions, max_length=300, padding='max_length', truncation=True,
                                return_tensors='pt').to(self.device)
        return self.text(tokens.input_ids, tokens.attention_mask, use_cache=False)[0], tokens.attention_mask.bool()


def null_metadata(cache):
    return dict(schema=1, source_text_cache_fingerprint=cache.fingerprint,
                null_template='raw-no-chi', negative_prompt=NEGATIVE_PROMPT)


def load_null_cache(root):
    reader = TextCacheReader(root)
    path = Path(root) / 'pixeldit_special.pt'
    if not path.exists():
        raise FileNotFoundError(f'{path}: run python -m pid._src.pixeldit_qdm.conditioning '
                                '--weights-root "$PID_WEIGHTS_ROOT" --text-cache-root "$PID_TEXT_CACHE_ROOT"')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['metadata'] != null_metadata(reader):
        raise ValueError('PixelDiT null cache differs from its caption cache or encoding template')
    for key in ('empty_embs', 'null_embs'):
        if value[key].shape != (1,300,2304):
            raise ValueError(f'Invalid PixelDiT null embedding shape: {key}')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root', default='weights')
    parser.add_argument('--text-cache-root', required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    reader = TextCacheReader(args.text_cache_root)
    path = Path(args.text_cache_root) / 'pixeldit_special.pt'
    if path.exists():
        load_null_cache(args.text_cache_root)
        print(f'Native PixelDiT null cache already complete: {path}')
        return
    encoder = PixelDiTTextEncoder(args.weights_root, 'cuda')
    embeddings, masks = encoder.encode_raw(['', NEGATIVE_PROMPT])
    value = dict(metadata=null_metadata(reader), empty_embs=embeddings[:1].cpu(), empty_mask=masks[:1].cpu(),
                 null_embs=embeddings[1:].cpu(), null_mask=masks[1:].cpu())
    temporary = path.with_suffix('.pt.incomplete')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    print(f'Native PixelDiT null cache complete: {path}; existing caption shards are unchanged')


if __name__ == '__main__':
    main()
