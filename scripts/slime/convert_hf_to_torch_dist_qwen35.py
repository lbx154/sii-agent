"""Compatibility launcher for slime Qwen3.5 HF -> Megatron conversion.

slime has Qwen3.5 Megatron specs and mbridge mappings, but the current
Transformers release in the slime env does not yet register `qwen3_5`.
Register minimal config classes before delegating to slime's converter.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig, PretrainedConfig


class Qwen35TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        rope_parameters = kwargs.get("rope_parameters") or {}
        if getattr(self, "rope_theta", None) is None:
            self.rope_theta = rope_parameters.get("rope_theta", 10000000)
        if getattr(self, "partial_rotary_factor", None) is None:
            self.partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 0.25)


class Qwen35Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, text_config: dict[str, Any] | Qwen35TextConfig | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            self.text_config = Qwen35TextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen35TextConfig()
        else:
            self.text_config = text_config

        for name in (
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "rms_norm_eps",
            "attention_dropout",
        ):
            if getattr(self, name, None) is None and getattr(self.text_config, name, None) is not None:
                setattr(self, name, getattr(self.text_config, name))


def _register_qwen35_configs() -> None:
    for model_type, config_cls in (
        ("qwen3_5_text", Qwen35TextConfig),
        ("qwen3_5", Qwen35Config),
    ):
        AutoConfig.register(model_type, config_cls, exist_ok=True)


def main() -> None:
    _register_qwen35_configs()
    repo_root = Path(__file__).resolve().parents[2]
    slime_dir = Path(os.getenv("SLIME_DIR", repo_root / "third_party" / "slime"))
    sys.path.insert(0, str(slime_dir))
    runpy.run_path(str(slime_dir / "tools" / "convert_hf_to_torch_dist.py"), run_name="__main__")


if __name__ == "__main__":
    main()
