"""Token-aligned capture accumulation across exact, stateful prefill chunks.

Only the raw input ids cover the whole request. Features before response_start-W
are discarded while capturing; retained chunks are packed before leaving GPU.
The final head row of one chunk supplies p for the first token of the next.
"""

import torch

from .data import pad_support, quantize_features


class CaptureSequence:
    def __init__(self, record, store, layer_ids, window=4096, encoding="fp8_e4m3fn"):
        self.record = record
        self.store = store
        self.layer_ids = tuple(layer_ids)
        if encoding not in ("fp8_e4m3fn", "bf16"):
            raise ValueError("unsupported capture encoding")
        self.encoding = encoding
        self.tokens = record["prompt_ids"] + record["output_ids"]
        self.start = max(0, len(record["prompt_ids"]) - window)
        self.offset = 0
        self.encoded = []
        self.scales = []
        self.target_ids = []
        self.target_probs = []
        self.previous_ids = self.previous_probs = None

    @torch.no_grad()
    def append(self, prefix, input_ids, features, next_ids, next_probs):
        tokens = input_ids.detach().cpu().long().tolist()
        length = len(tokens)
        if prefix != self.offset or length == 0:
            raise ValueError("capture chunks must be nonempty, contiguous and ordered")
        if tokens != self.tokens[prefix : prefix + length]:
            raise ValueError("captured token ids do not match the registered request")
        if (
            features.shape[:2] != (length, len(self.layer_ids))
            or next_ids.ndim != 2
            or next_ids.shape[0] != length
            or next_ids.shape[1] < 20
            or next_probs.shape != next_ids.shape
        ):
            raise ValueError("capture chunk shapes do not match token count/layers")
        width = max(
            next_ids.shape[-1],
            self.previous_ids.shape[-1] if self.previous_ids is not None else 20,
        )
        aligned_ids = torch.zeros(
            length, width, dtype=next_ids.dtype, device=next_ids.device
        )
        aligned_probs = torch.zeros(
            length, width, dtype=next_probs.dtype, device=next_probs.device
        )
        aligned_ids[1:, : next_ids.shape[-1]] = next_ids[:-1]
        aligned_probs[1:, : next_probs.shape[-1]] = next_probs[:-1]
        if self.previous_ids is not None:
            aligned_ids[0] = pad_support(self.previous_ids.to(next_ids.device), width)
            aligned_probs[0] = pad_support(
                self.previous_probs.to(next_probs.device), width
            )
        keep = max(0, self.start - prefix)
        if keep < length:
            if self.encoding == "bf16":
                raw, scales = features[keep:].cpu().bfloat16().clone(), None
            else:
                raw, scales = quantize_features(features[keep:])
            self.encoded.append(raw)
            if scales is not None:
                self.scales.append(scales)
            self.target_ids.append(aligned_ids[keep:].cpu().int().clone())
            self.target_probs.append(aligned_probs[keep:].cpu().float().clone())
        self.previous_ids = next_ids[-1].cpu().clone()
        self.previous_probs = next_probs[-1].cpu().clone()
        self.offset += length
        if self.offset < len(self.tokens):
            return None
        if self.offset != len(self.tokens):
            raise ValueError("capture extended beyond the registered request")
        width = max(ids.shape[-1] for ids in self.target_ids)
        return self.store.write_encoded(
            self.record,
            torch.cat(self.encoded),
            torch.cat(self.scales) if self.scales else None,
            torch.cat([pad_support(ids, width) for ids in self.target_ids]),
            torch.cat([pad_support(p, width) for p in self.target_probs]),
            start=self.start,
            encoding=self.encoding,
            layer_ids=self.layer_ids,
        )
