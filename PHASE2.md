# Phase 2 / 2A — Poincare Control Probe on Stock Llama-3.2-1B

A zero-touch, frozen-model control experiment.  We do **not** modify a single
weight, LoRA parameter, or projection matrix.  We attach **non-blocking
monitor hooks** at four depth checkpoints (`model.model.layers[4/8/12/15]`),
push every 2048-D hidden state onto the Poincare ball with the fixed spec
projection

```
proj(h) = h / (1 + sqrt(1 + ||h||^2))         R^2048 -> open unit ball
d_P(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
u (+) v  = ((1+2<u,v>+||v||^2)u + (1-||u||^2)v) / (1+2<u,v>+||u||^2||v||^2)
```

and log per-turn metrics while a real conversational REPL runs with a **real
KV cache** (`model.generate(use_cache=True)`).

This is the baseline data that lets Phase 3 *state why it works*: it measures
the stock manifold — at multiple depths — before any geometric constraint is
imposed, plus the effect of an **unlearned interior contraction**
(`h -> alpha*h`, `alpha ~ 0.85`) that pulls the boundary-pinned shell back
into the active volume of `B^2048`.

## The three control questions

1. **Native Manifold Drift** — where do layer-4/8/12/15 hidden states live,
   and how does each move across conversational turns before constraints?
2. **Working Memory Trajectory** — does a Poincare metric (and its guardian
   contraction) preserve/hierarchise context as the KV cache grows? (We only
   *measure* here — nothing is trained.)
3. **Ablation Baseline (Δ)** — a bit-reproducible, scripted stock metric
   stream (per monitored layer) that Phase 3's Cl(3,3) rotor sandwich engine
   runs with the *same harness* and diffs for the exact stability delta.

## Setup / files

| File | Role |
|------|------|
| `phase2_poincare_probe.py` | Frozen model load, 4 monitor hooks, fixed projection/`d_P`/Möbius math, `--script` + REPL, per-layer metrics, contraction check, JSONL/meta/PNG export. |
| `tests/demo_turns.json` | Deterministic scripted turn stream (3 turns). |
| `logs/phase2a_baseline.*` | Canvassed v2 run (greedy, layers 4/8/12/15, alpha 0.85). |
| `logs/phase2_baseline.*` | Historic v1 (layer 8 only) run, kept for continuity. |
| `bench_context_needle.py` | Context-needle stress bench (4K–64K prefill) over the same engine vs a hook-free stock baseline; JSONL/summary export. |
| `logs/context_needle.jsonl` | Official pre-intervention 4K–64K needle baseline (SCHEMA `bench.context_needle.v1`). |

## Snapshots and metrics (per monitored layer L)

- **`u_n^L`** (prompt-side): `proj(h_L)` of the **final prompt token** — the
  context-summary read *before* replying.
- **`a_n^L`** (reply-side): `proj(h_L)` of the **final generated token**.
- `T0^L = u_1^L` (turn 1, "Good morning.. what are you?").

Per turn, per layer: `prefill_radius.{min,max,mean,std,bounded}`, `u_radius`,
`a_radius`, `u_raw_norm`, `dP_T0_Tn`, `dP_prev`, `dP_walk`,
`cos_raw_T0_Tn`, `euclid_proj_T0_Tn`, `inflation` (= `dP_prev/euclid`),
`mobius_norm` (= `||(-u_{n-1}) (+) u_n||`, the geodesic velocity of the
working memory), plus contracted-stream twins `dP_T0_Tn_alpha`,
`dP_walk_alpha`.

**Contraction check** (directive 3), built from this turn's prefill radii at
every monitored layer:

```
contraction = { alpha: 0.85,
                r_max_raw:  max projected radius (boundary probe),
                r_max_alpha: alpha * r_max_raw,
                r_bos_raw:   radius of the pos-0 BOS token,
                r_bos_alpha: alpha * r_bos_raw,
                margin_from_boundary: 1 - r_max_alpha,
                active: r_max_alpha <= ACTIVE_R (0.9) }
```

**Causal pipeline self-test** per layer: every token of the turn-1 prompt must
be reproduced identically in later prefills.  Reported as the max per-position
RELATIVE error `drift_rel_prefix` (so the norm-800 BOS cannot mask genuine
differences) + absolute anchor error `drift_anchor_abs`.  Turn 1 is `None`
(no reference yet).

## What the verification runs show

All runs bit-reproducible across two identical scripted executions (md5-equal
`.jsonl` and `.png`).

### Directives 1+2 — depth gradient (turns of `demo_turns.json`, greedy)

| metric (turn 3) | L4 | L8 | L12 | L15 |
|---|---|---|---|---|
| `u_radius` | 0.726 | 0.781 | 0.882 | 0.953 |
| `dP_T0_Tn` | 2.553 | 3.022 | 4.313 | 6.290 |
| `dP_walk` (cum) | 4.243 | 5.375 | 7.829 | 11.716 |
| `cos_raw_T0_Tn` | 0.425 | 0.420 | 0.329 | 0.141 |
| `mobius_norm` (t2) | 0.841 | 0.909 | 0.973 | 0.997 |

**Finding A — curvature-regime gradient with depth.**  Deeper layers dwell
closer to the boundary (`u_radius` 0.73 → 0.95) and therefore inflate every
geodesic quantity: `dP_walk` grows 2.8x from L4 → L15, Möbius velocity ~1.0 at
L15 means memory steps are already in the high-curvature shell.  The final
layer (index 15, "depth 16") is the most boundary-pinned checkpoint of all —
exactly the layer whose geometry Phase 3's rotors must stabilise first.

**Finding B — the boundary shell is present at every depth.**  At all four
depths the BOS token (raw `||h_L|| ~ 800`) projects to `r ~ 0.9987` (0.9968 at
L15), while content tokens sit 0.71–0.92.  The stock manifold is a
nearly-isoradial shell hugging the boundary.

**Finding C — an unlearned `alpha = 0.85` clears the whole shell for free.**
`alpha*proj` maps BOS `0.9987 -> 0.8489` (L4/8/12) and `0.9968 -> 0.8473`
(L15): `margin_from_boundary >= 0.151`, `active=True` at **all** monitored
depths.  Every contraction check passed.  Caveat: a uniform, content-blind `α`
squeezes the *whole* manifold (killing dynamic range); Phase 3 replaces it
with a **learned, content-dependent interior `W_mem`** so the ball is used
actively without a global squeeze.  Drift self-test stays at fp16 noise
(`1.3e-3` → `2.4e-3` from L4 → L15), i.e. the instrument is clean and the
`dP`/`cos` deltas above are real.

## How to run

```bash
cd /home/bedroom_pc/Replace_KV_llama
PY=/home/bedroom_pc/miniconda3/envs/sandbox_env/bin/python

# interactive, four depth checkpoints (the classic Phase-2A session)
$PY phase2_poincare_probe.py --cache-dir llama32-1B-fp16 \
    --layers 4,8,12,16 --alpha 0.85 \
    --temperature 0.6 --top-p 0.9 --max-new-tokens 64

# deterministic baseline stream (greedy) -- the Phase 3 diff target
$PY phase2_poincare_probe.py --cache-dir llama32-1B-fp16 \
    --layers 4,8,12,16 --alpha 0.85 \
    --script tests/demo_turns.json --max-new-tokens 64 \
    --log-prefix logs/phase2a_baseline

# single-layer legacy behaviour (Phase 2 command still works)
$PY phase2_poincare_probe.py --cache-dir llama32-1B-fp16 --layer 8 \
    --temperature 0.6 --top-p 0.9 --max-new-tokens 64

# keep projected snapshots + Mobius vectors in the JSONL (~4*12 KB/turn)
$PY phase2_poincare_probe.py --cache-dir llama32-1B-fp16 \
    --script tests/demo_turns.json --save-vectors
```

`--layers` accepts a comma list; values `>= n_layers` clamp to the final layer
(so the spec's "16" monitors index 15).  `--alpha` defaults to `0.85`.
REPL meta-commands: `:reset`, `:metrics`, `:quit`.

## Schema (v2)

`SCHEMA = "phase2.probe.v2"`; one JSON line per turn in the `.jsonl`.
Top level: `schema, turn, user_text, reply, n_context, n_gen,
drift_rel_prefix (max over layers), drift_anchor_abs (max), layers{...}`.
Per-layer dict keys (string layer index): `prefill_radius, contraction,
u_radius, u_raw_norm, a_radius, dP_T0_Tn, dP_prev, dP_walk, dP_T0_Tn_alpha,
dP_walk_alpha, cos_raw_T0_Tn, euclid_proj_T0_Tn, inflation, mobius_norm,
drift_rel_prefix, drift_anchor_abs` (+ `u_proj/a_proj/mobius_vec` with
`--save-vectors`).  `.meta.json` carries the run header (`layers`, `alpha`,
seed, script...), `final` per-layer summary, and `q1/q2/q3` blocks —
`q3.diff_keys` names the exact fields Phase 3 compares.

## Context-needle stress baseline (pre-intervention)

The Phase-3 comparison matrix at long prefill: a 2-turn doc-read/query loop over
the **same verified engine** (`phase2_poincare_probe.py`) at 4K / 16K / 32K /
64K prompt tokens, against a hook-free stock decode of identical input.

```
SCHEMA = "bench.context_needle.v1"        (logs/context_needle.jsonl)
needle   = 'The magic number is 42813.'    buried at depth 0.5 in a
           3-paragraph filler document (rotating tiles)
protocol = turn 1: "Read the following document carefully, then respond with
           exactly 'OK'."   |   turn 2: "Using only the document I just gave
           you, what is the magic number? Reply with just the digits."
decode   = greedy; doc max 8 new tokens; answer max 24, min 16 (min_new_tokens
           forces a non-empty span: stock 1B logits collapse to an immediate
           <|eot_id|> after long context dumps, so forced spans keep the
           retrieval score a fair poincare-vs-stock comparison)
```

Reproduce:

```bash
PY=/home/bedroom_pc/miniconda3/envs/sandbox_env/bin/python
$PY bench_context_needle.py --cache-dir llama32-1B-fp16 --layers 4,8,12,16 \
    --alpha 0.85 --sizes 4096,16384,32768,65536 --needle-depth 0.5 \
    --doc-max-new-tokens 8 --ans-max-new-tokens 24 --min-answer-tokens 16 \
    --seed 0 --log-prefix logs/context_needle
```

### Sweep results (needle depth 0.5)

| size | mode | retrieved | peakMB | wall_s | n_ctx | dP L15 | walk L15 | cos L15 | mob L15 | bounded |
|------|------|:---------:|-------:|-------:|------:|-------:|---------:|--------:|--------:|:-------:|
| 4096 | poincare | F | 2940.7 | 2.98 | 4154 | 4.05 | 4.05 | 0.963 | 0.966 | yes |
| 4096 | stock | F | 2889.1 | 0.99 | — | — | — | — | — | n/a |
| 16384 | poincare | F | 4296.8 | 5.15 | 16381 | 3.53 | 3.53 | 0.980 | 0.943 | yes |
| 16384 | stock | F | 4095.4 | 4.39 | — | — | — | — | — | n/a |
| 32768 | poincare | F | 6106.5 | 27.40 | 32673 | 3.04 | 3.04 | 0.990 | 0.909 | yes |
| 32768 | stock | F | 5704.3 | 25.47 | — | — | — | — | — | n/a |
| 65536 | poincare | F | 9718.0 | 92.51 | 65259 | 3.56 | 3.56 | 0.982 | 0.945 | yes |
| 65536 | stock | F | 8916.0 | 73.02 | — | — | — | — | — | n/a |

retrieval acc[poincare]: 4096:0%, 16384:0%, 32768:0%, 65536:0%
retrieval acc[stock]:    4096:0%, 16384:0%, 32768:0%, 65536:0%

Per-layer geodesic state at the 64K point (turn 2, poincare):

| layer | dP_T0_Tn | dP_walk | cos_raw_T0_Tn | mobius_norm | R_max | bounded | contraction_active | r_bos_alpha |
|-------|---------:|--------:|--------------:|------------:|------:|:-------:|:-----------------:|------------:|
| 4 | 0.579 | 0.580 | 0.997 | 0.282 | 0.999 | yes | yes | 0.849 |
| 8 | 0.442 | 0.443 | 0.996 | 0.217 | 0.999 | yes | yes | 0.849 |
| 12 | 1.794 | 1.796 | 0.982 | 0.715 | 0.999 | yes | yes | 0.849 |
| 15 | 3.558 | 3.560 | 0.982 | 0.945 | 0.997 | yes | yes | 0.847 |

**Finding N1 — retrieval is 0% for *both* modes at every size.**  Stock
Llama-3.2-1B-Instruct cannot extract the needle from filler context; independent
of format (RAG-style framing, real-corpus filler, sampling, needle at depth
0.9) it either emits an immediate `<|eot_id|>` or degenerates after ~400+ token
context dumps.  The 0/0 split is the honest pre-intervention subtraction point
for Phase 3 to beat.

**Finding N2 — geodesic stability holds to 64K under the monitored engine.**
`bounded=True` and contraction `active=True` at all four depths and all four
scales; the α-contraction anchor stays at the expected `r_bos_alpha` 0.847–0.849.
The depth gradient (shallow near-static `cos 0.997` → deep dynamic L15) survives
the full 64K range.

**Finding N3 — the deep-layer geodesic is non-monotonic across scale.**  L15
`dP_T0_Tn` runs 4.05 → 3.53 → 3.04 → 3.56 (4K→64K), `cos` gathers 0.963 → 0.990
then relaxes to 0.982, and Möbius velocity dips to 0.909 at 32K.  The manifold's
scale response is *not* a flat curve — Phase 3 should target this exactly.

**Finding N4 — monitored overhead is small.**  Peak allocation vs stock:
+51.6 MB @4K, +201.4 MB @16K, +402.2 MB @32K, +802 MB @64K (absolute 9718 MB).
The observation layer costs seconds of wall time (2.98 s → 92.5 s at 64K).

## Verified environment

llama32-1B-fp16 on GPU (fp16, frozen, 1.236B) · transformers 5.17.0 · torch
2.14 CUDA · RTX 4060 Ti 8 GB · 4 hooks + 2 scripted runs bit-identical.