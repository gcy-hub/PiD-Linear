"""The frozen Gemma condition encoder, shared by online and offline encoding."""

from pathlib import Path

import torch

from pid._src.configs.pid.experiment_2kto4k_v1pt5.shared_config import _CHI_PROMPT

TEXT_LENGTH = 300
TEXT_DIM = 2304
PROMPT_PREFIX = "\n".join(_CHI_PROMPT)
NEGATIVE_PROMPT = "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, watermark"


class GemmaTextEncoder:
    def __init__(self, weights_root, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(weights_root) / "gemma-2-2b-it"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.text = (
            AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, local_files_only=True)
            .get_decoder()
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.prefix_tokens = len(self.tokenizer.encode(PROMPT_PREFIX))

    @torch.no_grad()
    def encode_text(self, captions):
        tokens = self.tokenizer(
            [PROMPT_PREFIX + caption for caption in captions],
            max_length=self.prefix_tokens + 298,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        embs = self.text(tokens.input_ids, tokens.attention_mask, use_cache=False)[0]
        indices = [0] + list(range(-299, 0))
        return embs[:, indices], tokens.attention_mask[:, indices].bool()
