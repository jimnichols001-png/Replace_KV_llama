# Phase 2A: Poincare Ball State Visualization & Context-Window Responsiveness

## Summary

Built a visualization and testing pipeline (`vis_poincare_ball.py`) that captures the
full projected point cloud from the Poincare-ball memory engine at each context size
(4K→16K→32K→64K), producing:

- **Radius-vs-position profiles** per monitored layer
- **2D disk PCA views** (unit-disk geometry preserved)
- **4-turn responsiveness/forgetfulness scenario**: doc→query→distractor→requery

## Context-Window Test Results

### Retrieval (needle = "42813")

| Turn | 4K | 16K | 32K | 64K |
|------|----|----|-----|-----|
| query | OK | OK | OK | OK |
| requery (after distractor) | OK | OK | OK | OK |

**100% needle retrieval across 4K–64K context. Zero forgetfulness.**

### Poincare Ball Geometry (projected radius max, all bounded < 1)

| Layer | 4K | 16K | 32K | 64K |
|-------|-----|------|------|------|
| L4  | 0.72 | 0.83 | 0.83 | 0.83 |
| L8  | 0.78 | 0.87 | 0.87 | 0.87 |
| L12 | 0.89 | 0.93 | 0.93 | 0.93 |
| L15 | 0.95 | 0.97 | 0.97 | 0.97 |

Radii plateau across sizes — the geometry is stable, not degrading with context.

### Performance

| Size | Wall time | Peak GPU |
|------|-----------|----------|
| 4K   | 5s        | ~3 GB    |
| 16K  | 5s        | 4.3 GB   |
| 32K  | 27s       | 6.1 GB   |
| 64K  | 114s      | 9.7 GB   |

## SDPA GQA Regression Fix

### Root Cause

torch 2.5.1+cu124 on Windows: SDPA with `enable_gqa=True` during prefill (q_len > 1)
silently falls back to the fp32 **math** backend, materializing
`[n_heads, seq, seq] float32` attention (~32 GiB at 16K context). This OOMs any
context beyond ~5K tokens. Decode (q_len=1) is unaffected.

The model reaches this path because:
1. transformers 5.12 passes `attention_mask=None` to SDPA layers for SDPA attn
2. `use_gqa_in_sdpa(None, key)` returns True → `enable_gqa=True` → math fallback

### Fix

Patch `use_gqa_in_sdpa` to always return False, forcing the repeat-kv expansion
(KV heads 8→32) and the memory-bounded flash/mem_efficient kernels:

```python
# phase2_poincare_probe.py (module-level, import-time)
def _disable_gqa_in_sdpa():
    try:
        import transformers.integrations.sdpa_attention as _sdpa
        _sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
    except Exception:
        pass

_disable_gqa_in_sdpa()
```

### Verification

| Config | Prefill peak (16K seq) | Status |
|--------|----------------------|--------|
| enable_gqa=True (default) | 31.84 GiB (OOM) | FAIL |
| repeat-kv expanded (patched) | 0.28 GB | OK |

## Output Files

- `logs/ball_20260914_154841.jsonl` — per-turn JSONL with radii/geodesics
- `logs/ball_20260914_154841_points.npz` — point clouds + PCA2D + radii per size
- `logs/ball_20260914_154841.png` — cross-size summary (4 panels)
- `logs/ball_20260914_154841/{radius,disk}_{4096,16384,32768,65536}.png` — per-size plots

## Files Changed

- **`vis_poincare_ball.py`** — new: visualizer + 4-turn responsiveness/forgetfulness test
- **`phase2_poincare_probe.py`** — added `_disable_gqa_in_sdpa()` workaround
