# Backbone


See the design and implementation history under `docs/superpowers/`:
- `specs/2026-06-27-backbone-design.md` — Spec 1, initial codebase shell
- `plans/2026-06-27-backbone.md` — Spec 1 implementation plan
- `specs/2026-06-28-vortex-fidelity-architecture-design.md` — Spec 2, vortex-faithful SH2 architecture rewrite
- `plans/2026-06-28-vortex-fidelity-architecture.md` — Spec 2 implementation plan
- `specs/2026-06-29-epitope-gen-design.md` — Spec 3, guided flow matching for neoantigens
- `plans/2026-06-29-epitope-gen-phase1-foundation.md` — Spec 3 / Phase 1 plan (data + ESM-2)

## Install

    make install

## Test

    make test

## Generate (random-init smoke)

    python generate.py \
      --config configs/backbone-small-50m.yml \
      --prompt "<5UTR>ACGU" \
      --max-new-tokens 8 \
      --device cpu --dtype float32 --tiny-debug

## Configs

- `configs/backbone-small-50m.yml`  — 50M params
- `configs/backbone-base-200m.yml`  — 200M params (default for `run_generate`)
- `configs/backbone-large-1b.yml`   — 1B params

All three use a 16-token mRNA vocab and Convolutional Multi-Hybrid Architecture: four block kinds (`attn`, `hcs`, `hcm`, `hcl`) assigned to
layer indices via `attn_layer_idxs` / `hcs_layer_idxs` / `hcm_layer_idxs` /
`hcl_layer_idxs`. The convolutional multi-hybrid lives inside `HyenaCascade`
(3-projection front FIR → column-split → inner filter → gate → out
projection), wrapped by `ParallelGatedConvBlock`. See
`docs/superpowers/specs/2026-06-28-vortex-fidelity-architecture-design.md`
for the full design.

## Out of scope (in this `backbone` package)

Training and the SH2 backbone's guided generation are not in this
package. Epitope generation (guided flow matching, MHC-binding heads,
mRNA hand-off) lives in the parallel `epitope_gen/` package.
