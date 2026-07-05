"""CLI entry point — load a config, instantiate a model (random init if no
checkpoint), tokenize a prompt, generate, print.
"""
import argparse
import sys
from pathlib import Path

import torch

from backbone.model.engine import BackboneEngine
from backbone.mrna.regions import parse_construct


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--parse-construct", action="store_true")
    p.add_argument(
        "--tiny-debug",
        action="store_true",
        help="Shrink config in-memory for fast CPU smoke tests.",
    )
    return p.parse_args()


def _maybe_tiny(cfg):
    cfg.hidden_size = 32
    cfg.num_layers = 4
    cfg.num_attention_heads = 4
    cfg.num_kv_heads = 2
    cfg.intermediate_size = 64
    cfg.max_seqlen = 64
    cfg.state_size = 32
    cfg.num_filters = 32
    cfg.attn_layer_idxs = [3]
    cfg.hcl_layer_idxs = [2]
    cfg.hcm_layer_idxs = [1]
    cfg.hcs_layer_idxs = [0]
    cfg.short_filter_length = 3
    cfg.hcs_filter_length = 3
    cfg.hcs_filter_groups = 32
    cfg.hcm_filter_length = 8
    cfg.hcm_filter_groups = 8
    cfg.hcl_filter_order = 8
    cfg.hcl_filter_groups = 8
    return cfg


def main() -> int:
    args = _parse_args()
    if args.prompt is None and args.prompt_file is None:
        print("--prompt or --prompt-file required", file=sys.stderr)
        return 2
    prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text().strip()

    if args.tiny_debug:
        from backbone.model.config import BackboneConfig
        real = BackboneConfig.from_yaml

        def _patched(path):
            return _maybe_tiny(real(path))

        BackboneConfig.from_yaml = staticmethod(_patched)  # type: ignore[assignment]

    engine = BackboneEngine.from_pretrained(
        args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    result = engine.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    text = engine.tokenizer.decode(result.token_ids, skip_special_tokens=False)
    print(text)
    if args.parse_construct:
        try:
            c = parse_construct(text)
            print(f"# construct: 5utr={len(c.five_utr or '')}, cds={len(c.cds or '')}, "
                  f"3utr={len(c.three_utr or '')}, polyA={c.polyA_len or 0}")
        except ValueError as e:
            print(f"# parse failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
