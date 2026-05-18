"""Create a SGLang-compatible subset of a Qwen3.5 LoRA adapter.

SGLang 0.5.11 can serve Qwen3.5 base weights, but its LoRA path currently
does not support the linear-attention adapter modules in_proj_a/b/qkv/z and
out_proj. This utility keeps only the supported attention/MLP adapter tensors
so the server can load the adapter for runtime experiments.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SUPPORTED_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source PEFT LoRA adapter directory.")
    parser.add_argument("--dst", required=True, help="Destination adapter directory.")
    parser.add_argument("--base-model", default="/root/sii-agent/Qwen3.5-9B")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    tensors = {}
    with safe_open(src / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            parts = key.split(".")
            module = parts[-3] if len(parts) >= 3 and parts[-2].startswith("lora_") else ""
            if module in SUPPORTED_MODULES:
                tensors[key] = handle.get_tensor(key)
    if not tensors:
        raise RuntimeError(f"No supported LoRA tensors found in {src}")
    save_file(tensors, dst / "adapter_model.safetensors")

    config = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = sorted(SUPPORTED_MODULES)
    config["base_model_name_or_path"] = args.base_model
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = src / "README.md"
    if readme.exists():
        shutil.copy2(readme, dst / "README.md")
    print(f"Wrote {len(tensors)} tensors to {dst}")


if __name__ == "__main__":
    main()
