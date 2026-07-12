# Antigenify

A convolutional multi-hybrid sequence model for mRNA generation. Pure PyTorch,
runnable on CPU, sized from 50M to 1B parameters.

## What "convolutional multi-hybrid" means here

Sequence mixers in transformers all attend over the full context. This model
stripes different mixer types across layers, and inside the convolutional
mixer chains three filter stages together — hence *convolutional multi-hybrid*.

Two block kinds share the model:

| Block | Mixer | What it's good at |
|---|---|---|
| `AttentionBlock` | grouped-query self-attention with rotary positions | content-addressable long-range interactions |
| `ParallelGatedConvBlock` | `HyenaCascade` (see below) | local + medium + long-range structure at conv cost |

Layer assignment is driven by four disjoint index lists in the config:

```yaml
attn_layer_idxs: [3, 7, 11, 15, 19, 23]
hcl_layer_idxs: [2, 6, 10, 14, 18, 22]
hcm_layer_idxs: [1, 5, 9, 13, 17, 21]
hcs_layer_idxs: [0, 4, 8, 12, 16, 20]
```

Every layer index in `range(num_layers)` appears in exactly one list. The
three `hc*` categories all instantiate `ParallelGatedConvBlock`; they differ
only in how the cascade's *inner filter* is parametrized (see below).

### HyenaCascade

`HyenaCascade` (in `backbone/model/model.py`) is a single operator that runs
input through three stages plus a gate:

```
x  →  short FIR (3 projections: x1, x2, v)   ← depthwise causal conv, K = 7
       |
       column-split into (x1, x2, v) of shape (B, D, T)
       |
       (x2 * v)  →  inner filter  →  inner_out
                    (hcs / hcm / hcl)
       |
       gated = x1 * inner_out    (+ D * v  if inner filter length ≥ 128)
       |
       out_proj(gated)
```

The inner filter has three parametrizations, selected by the block's category:

| Category | Inner filter | Typical length |
|---|---|---|
| `hcs` | explicit short depthwise FIR | ~7 |
| `hcm` | explicit medium depthwise FIR | ~128 |
| `hcl` | implicit filter — a tiny MLP over learned positional embeddings maps position → per-group filter coefficient; applied via causal FFT convolution | full context |

The `hcl` variant is what lets the model capture unbounded-range structure
at conv cost: the filter length grows with the sequence, but the filter
*parameters* remain fixed (an MLP of size `hcl_filter_order`).

The optional skip-connection `D` (a learned per-channel bypass around the
inner filter) activates when the inner filter is long enough (≥ 128) to
benefit from a direct pass-through.

### Attention half

The `AttentionBlock` uses `MHA` — grouped-query self-attention (fused QKV
projection, rotary positional embeddings, causal SDPA) sharing the same
`ParallelGatedMLP` and `RMSNorm` conventions as the conv block. Optional
FlashAttention-2 and ALiBi paths are available on CUDA; the default
pure-PyTorch path runs on CPU.

### Cache

Both block families share a single `Cache` object during step-by-step
generation:

- Attention layers store per-layer `(k, v)` tensors that grow with each
  token.
- Conv layers store *two* pieces of state per layer: the last `K-1` tokens
  fed into the front FIR, and either the last `L-1` tokens fed into the
  inner filter (for `hcs`/`hcm`) or the full prefix (for `hcl`, since its
  filter is unbounded).

The load-bearing correctness check is that step-by-step generation with
this cache produces logits identical (to `atol=1e-3`) to a full-sequence
forward pass on the same prefix. This test lives in `test/test_cache.py`
and exercises a mixed-block tiny model covering all four kinds.

## Vocab and construct format

The tokenizer (`backbone/model/tokenizer.py`) uses a 16-token vocabulary
tuned for mRNA construct prompts:

| Range | Tokens |
|---|---|
| 0–3 | `<pad>`, `<bos>`, `<eos>`, `<unk>` |
| 4–8 | `A`, `C`, `G`, `U`, `N` |
| 9–14 | Paired region tags: `<5UTR>...</5UTR>`, `<CDS>...</CDS>`, `<3UTR>...</3UTR>` |
| 15 | `<polyA>` (marker — the tail is encoded as trailing `A` tokens) |

Region tags tokenize as single IDs (greedy longest-match), so
`<5UTR>ACGU</5UTR>` becomes 3 tokens, not 12. `T` is normalized to `U` by
default so DNA prompts work transparently.

Small helper module: `backbone/mrna/regions.py` provides `build_prompt(...)`
and `parse_construct(...)` for round-tripping structured constructs.

## Install

```
make install
```

## Test

```
make test
```

CPU-only runs cover the pure-PyTorch paths. CUDA + FlashAttention paths are
CUDA-gated and skip cleanly on CPU.

## Generate (random-init smoke)

```
python generate.py \
  --config configs/backbone-small-50m.yml \
  --prompt "<5UTR>ACGU" \
  --max-new-tokens 8 \
  --device cpu --dtype float32 --tiny-debug
```

The `--tiny-debug` flag shrinks the config in memory so the smoke test
runs in seconds without loading a real checkpoint. Without a checkpoint,
weights are random and the emitted sequence is not meaningful — a warning
prints to make this obvious.

## Configs

| File | Params | Layers | Hidden | Heads (Q/KV) | Context |
|---|---|---|---|---|---|
| `configs/backbone-small-50m.yml` | ~50M | 16 | 512 | 8 / 2 | 32k |
| `configs/backbone-base-200m.yml` | ~200M | 24 | 1024 | 16 / 4 | 32k |
| `configs/backbone-large-1b.yml` | ~1B | 32 | 2048 | 32 / 8 | 32k |

All three use the same stripe pattern (`[hcs, hcm, hcl, attn]` repeated),
the same 16-token mRNA vocab, and the same rotary base.

## Repository layout

```
backbone/
├── model/
│   ├── config.py          BackboneConfig dataclass, YAML I/O, layer-idx validation
│   ├── model.py           HyenaCascade, AttentionBlock, ParallelGatedConvBlock, StripedHyena, get_block
│   ├── layers.py          RMSNorm, ParallelGatedMLP, Embedding
│   ├── attention.py       MHA, SelfAttention/CrossAttention (pure-PyTorch),
│   │                       FlashSelf/CrossAttention (CUDA), get_alibi_slopes
│   ├── rotary.py          RotaryEmbedding, apply_rotary_emb_torch
│   ├── cache.py           KV + per-cascade conv-state cache
│   ├── engine.py          Model loader + generate wrapper
│   ├── generation.py      Autoregressive decode loop
│   ├── sample.py          temperature / top-k / top-p sampler
│   └── tokenizer.py       MRNATokenizer
├── mrna/
│   ├── alphabet.py        Alphabet constants + construct-string validator
│   └── regions.py         build_prompt / parse_construct / Construct dataclass
└── ops/                   Kernel implementations for optional CUDA paths
tokenizer/mrna_v1/         vocab.json + special_tokens.json
configs/                   3 size YAMLs
generate.py                CLI entry point
```
