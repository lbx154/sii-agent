"""Launch slime train.py with Qwen3.5 config registration.

The slime environment can train Qwen3.5 through its Megatron plugins, but the
installed Transformers release does not register the Qwen3.5 AutoConfig.  This
launcher registers a minimal config before delegating to slime's train entrypoint.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from scripts.slime.convert_hf_to_torch_dist_qwen35 import Qwen35Config, _register_qwen35_configs


def _qwen35_config_from_pretrained(original_from_pretrained: Any):
    def patched_from_pretrained(pretrained_model_name_or_path: str | os.PathLike[str], *args: Any, **kwargs: Any):
        config_path = Path(pretrained_model_name_or_path) / "config.json"
        if config_path.is_file():
            try:
                config = Qwen35Config.get_config_dict(str(pretrained_model_name_or_path), **kwargs)[0]
            except Exception:
                config = None
            if isinstance(config, dict) and config.get("model_type") == "qwen3_5":
                return Qwen35Config.from_dict(config, **kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    return patched_from_pretrained


def main() -> None:
    _register_qwen35_configs()
    repo_root = Path(__file__).resolve().parents[2]
    slime_dir = Path(os.getenv("SLIME_DIR", repo_root / "third_party" / "slime"))
    sys.path.insert(0, str(slime_dir))

    # Importing slime arguments imports SGLang, which registers its own Qwen3.5
    # AutoConfig with MoE defaults that are wrong for dense Qwen3.5-9B.
    from slime.utils.arguments import parse_args

    _register_qwen35_configs()
    original_from_pretrained = AutoConfig.from_pretrained
    AutoConfig.from_pretrained = staticmethod(_qwen35_config_from_pretrained(original_from_pretrained))
    try:
        args = parse_args()
    finally:
        AutoConfig.from_pretrained = original_from_pretrained

    train_globals = runpy.run_path(str(slime_dir / "train.py"), run_name="slime_train_entry")
    train_globals["train"](args)


if __name__ == "__main__":
    main()
