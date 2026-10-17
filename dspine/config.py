"""Serializable architecture contract, independent of a target model class."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class DSpineConfig:
    hidden_size: int = 2560
    vocab_size: int = 248320
    num_hidden_layers: int = 6
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 10240
    target_layer_ids: tuple[int, ...] = (1, 12, 23, 34, 45)
    target_hidden_size: int = 2560
    eos_token_ids: tuple[int, ...] = (248046, 248044)
    hc_count: int = 4
    block_size: int = 16
    context_window: int = 4096
    code_size: int = 1024
    message_size: int = 512
    candidate_count: int = 16
    rope_theta: float = 10000000.0
    rms_norm_eps: float = 1e-6
    feature_scale: float | None = None  # None means sqrt(code_size), paper §4.1.
    mask_token_id: int | None = None  # Legacy Tencent checkpoint uses a frozen row.
    gradient_checkpointing: bool = False
    context_fusion: str = "off"  # Zero-initialized previous-state input shortcut.

    def __post_init__(self):
        self.target_layer_ids = tuple(self.target_layer_ids)
        self.eos_token_ids = tuple(self.eos_token_ids)
        if not isinstance(self.context_fusion, str) or self.context_fusion not in {
            "off",
            "anchor",
            "all",
        }:
            raise ValueError("context_fusion must be off, anchor or all")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by kv heads")
        if self.head_dim % 2 or self.block_size < 2:
            raise ValueError("even head_dim and block_size >= 2 required")
        if (
            min(
                self.hidden_size, self.code_size, self.message_size, self.context_window
            )
            < 1
        ):
            raise ValueError("dimensions and window must be positive")

    @property
    def scale(self):
        return self.code_size**0.5 if self.feature_scale is None else self.feature_scale

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, obj):
        if "dflash_config" not in obj:
            return cls(
                **{k: v for k, v in obj.items() if k in {f.name for f in fields(cls)}}
            )
        d = obj["dflash_config"]
        b = d["selected_prefix_code_bridge"]
        return cls(
            hidden_size=obj["hidden_size"],
            target_hidden_size=obj["hidden_size"],
            vocab_size=obj["vocab_size"],
            num_hidden_layers=obj["num_hidden_layers"],
            num_attention_heads=obj["num_attention_heads"],
            num_key_value_heads=obj["num_key_value_heads"],
            head_dim=obj["head_dim"],
            intermediate_size=obj["intermediate_size"],
            target_layer_ids=tuple(d["target_layer_ids"]),
            hc_count=1,
            eos_token_ids=(obj["eos_token_id"],)
            if isinstance(obj["eos_token_id"], int)
            else tuple(obj["eos_token_id"]),
            block_size=obj["block_size"],
            code_size=b["code_size"],
            message_size=b["rank"],
            rope_theta=obj["rope_theta"],
            rms_norm_eps=obj["rms_norm_eps"],
            mask_token_id=d["mask_token_id"],
        )
