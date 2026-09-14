"""
vis_poincare_ball.py -- watch the Poincare ball "fill" as the context window grows,
and score responsiveness / forgetfulness at every scale.

Motivation
----------
The Phase 2A probe treats every token's hidden state, at each monitored depth, as
a point written into the Poincare ball B^2048:
        proj(h) = h / (1 + sqrt(1 + ||h||^2))
Per turn we currently only persist aggregates (prefill radius stats + the u_n / a_n
anchor points).  This script captures the FULL projected point cloud per monitored
layer at every turn, and produces:

  1.  Radius-vs-position profile   -- for each layer: token position (normalized to
      the growing context) vs projected radius.  This IS the storage media: you see
      the BOS pin at the boundary shell (r ~ 0.998), the content band
      (r ~ 0.71-0.92), and how the ball's shell grows as context -> 4K/16K/32K/64K.
  2.  2D ball-disk view (PCA)      -- principal-2-plane embedding of the point cloud,
      clipped to the unit disk, colored by token position, boundary r=1 + active
      r=0.9 rings.  The literal picture of what is stored where.
  3.  Responsiveness/forgetfulness -- a 4-turn scenario per context size:
        turn "doc"   : read a sized document (needle buried at needle-depth)
        turn "query" : retrieve the needle            -> responsiveness at size
        turn "dist": unrelated one-liner              -> interference
        turn "requery": re-retrieve the needle        -> forgetfulness under
                         interference, and does the ball retain the planted point?
      Each turn also keeps the per-layer geodesic stream (dP, walk, cos, Mobius),
      so the ball metrics stay diffable against Phase 3.

Outputs (log-prefix "logs/ball")
    <prefix>_<ts>.jsonl            one JSON line per turn: metrics + retrieval
    <prefix>_<ts>_points.npz       per (layer, turn): radius-per-position + PCA2D
    <prefix>_<ts>_size_<S>.png     per size: 2xN figure (radius profile | disk view)
    <prefix>_<ts>.png              cross-size summary (retrieval + L15 geodesics +
                                   planted-needle radius vs context length)

Guarantees: no weights touched, hooks are the same non-blocking monitors; the point
cloud is taken from the live hook capture AFTER each run_turn (nothing else changes
the probe).  Point capture / plotting failures never kill the metric run.

Usage
-----
    python vis_poincare_ball.py --cache-dir llama32-1B-Instruct-bf16 \
        --layers 4,8,12,16 --alpha 0.85 \
        --sizes 4096,16384,32768,65536 --needle-depth 0.5

    # fast smoke (checks plumbing, small sizes)
    python vis_poincare_ball.py --sizes 1024,4096
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

import phase2_poincare_probe as P2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

SCHEMA = "vis.poincare_ball.v1"
FILLER_PARAS = [
    "Perennial plant life along the river corridor sustains a diverse "
    "community of birds, small mammals, and insects. Seasonal flooding "
    "refreshes the soil and keeps the canopy vigorous throughout the warmer "
    "months. Field biologists catalog the species each autumn before the "
    "water level rises and the low meadows turn to marsh.",
    "The mining town was rebuilt twice in fifty years, first after the timber "
    "crisis and again when the rail line was rerouted through the northern "
    "pass. Its granite clock tower still keeps time, though few residents "
    "remember the original builders who cut and hauled every stone by hand.",
    "Researchers at the coastal observatory record sea temperatures at dawn "
    "and dusk, watching for the slow drift of currents that carries warm "
    "surface water toward the kelp forests. Their instruments are anchored "
    "to concrete blocks and must survive waves up to eight meters during "
    "the late-summer storms. Data is archived nightly at a secondary site "
    "for safety.",
]
FILLER = "".join(FILLER_PARAS)
NEEDLE_DEFAULT = "The magic number is 42813."
DOC_TURN = ("Read the following document carefully, then respond with "
            "exactly 'OK'.")
QUERY_TURN = ("Using only the document I just gave you, what is the magic "
              "number? Reply with just the digits.")
DISTRACTOR_TURN = "Tell me a fun fact about the ocean in one sentence."
REQUERY_TURN = ("What was the magic number from the document I gave you "
                "earlier? Reply with just the digits.")
TURNS = [
    ("doc", DOC_TURN),
    ("query", QUERY_TURN),
    ("distractor", DISTRACTOR_TURN),
    ("requery", REQUERY_TURN),
]


def build_doc(tokenizer, size, fraction, needle_text):
    """A document of ~`size` tokens with the needle at depth `fraction`."""
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    pool = tokenizer.encode(FILLER, add_special_tokens=False)
    if len(needle_tokens) + 32 >= size:
        raise ValueError(f"size {size} too small for needle "
                         f"({len(needle_tokens)} tokens)")
    pool_tiles = pool * (1 + size // max(1, len(pool)))
    prefix = int(size * fraction)
    tokens = (pool_tiles[:prefix] + needle_tokens +
              pool_tiles[prefix: size - len(needle_tokens)])
    return tokenizer.decode(tokens[:size])


def extract_points(probe, L):
    """Full projected point cloud [P, d] from the live hook capture at layer L.

    Must be called immediately after probe.run_turn (the next call clears the
    capture).  Mirrors the split used by DepthProbe._summarise_capture."""
    chains = probe.captured[L]
    prefill = [c for c in chains if c.shape[1] > 1]
    if prefill:
        pf = prefill[0].float()[0]
    else:
        decode = [c for c in chains if c.shape[1] == 1]
        pf = torch.cat(decode, dim=1).float()[0]
    return P2.project(pf)


def pca2d(pts, max_fit=2048, seed=0):
    """Project the point cloud onto its top-2 principal components.

    PCA is fit on a subsample (max_fit) for speed at 64K context, then every
    point is embedded.  Returns [N, 2] float32.

    The embedding is applied WITHOUT mean-subtraction (orthogonal directions
    only), so each embedded point keeps ||emb|| <= ||pt|| < 1 -- the unit disk
    in the figure is the genuine Poincare-ball boundary, not a data centroid."""
    n, d = pts.shape
    if n < 2 or d < 2:
        return torch.zeros(n, 2, dtype=torch.float32)
    if max_fit and n > max_fit:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_fit]
        fit = pts[idx]
    else:
        fit = pts
    decentered = fit - fit.mean(dim=0, keepdim=True)
    try:
        Vh = torch.linalg.svd(decentered, full_matrices=False)[2][:2]
    except Exception:
        return torch.zeros(n, 2, dtype=torch.float32)
    return pts @ Vh.t()


def needle_pos(prompt_ids, probe, needle_text):
    """Token index of the needle inside the (tokenized) prompt, or None."""
    mt = probe.tokenizer.encode(needle_text, add_special_tokens=False)
    if not mt:
        return None
    ids = prompt_ids[0].tolist()
    k = len(mt)
    for s in range(len(ids) - k + 1):
        if ids[s:s + k] == mt:
            return s
    return None


def run_turn(probe, track, pf_turn1, user_text, gen_kwargs):
    """Stateful turn through the Phase 2A engine (mirrors bench_context_needle);
    returns (reply, row) and leaves the live hook capture for point extraction."""
    track.add_user(user_text)
    reply, summ = probe.run_turn(track.conv, gen_kwargs)
    track.add_assistant(reply)
    drift = {}
    for L in probe.monitored:
        key = str(L)
        if pf_turn1[key] is not None:
            cur = summ[key]["pf_full"]
            ref = pf_turn1[key]
            Ln = ref.shape[0]
            ref_n = ref.norm(dim=-1).clamp_min(1e-6)
            delta = (cur[:Ln] - ref).abs()
            rel = (delta / ref_n.unsqueeze(-1)).max(dim=-1).values
            drift[L] = (float(rel.max().item()), float(delta[-1].max().item()))
        else:
            pf_turn1[key] = summ[key]["pf_full"]
    row = track.record(user_text, reply, summ, drift=drift)
    return reply, row


def json_safe(obj):
    import math
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def layer_metrics(row):
    out = {}
    for k, e in row.get("layers", {}).items():
        out[str(k)] = {
            "dP_T0_Tn": e["dP_T0_Tn"], "dP_prev": e["dP_prev"],
            "dP_walk": e["dP_walk"], "mobius_norm": e["mobius_norm"],
            "cos_raw_T0_Tn": e["cos_raw_T0_Tn"],
            "prefill_radius_max": e["prefill_radius"]["max"],
            "bounded": e["prefill_radius"]["bounded"],
            "contraction_active": e["contraction"]["active"],
            "r_bos_alpha": e["contraction"]["r_bos_alpha"],
        }
    return out


def radius_profile_figure(records_by_lookup, layers, turns, size, path, needle_positions):
    """Radius-vs-position profile: rows = layers, one line per turn."""
    if not HAS_MPL:
        return
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(13, 3.2 * n_layers),
                             squeeze=False)
    colors = dict(doc="#1f77b4", query="#2ca02c",
                  distractor="#ff7f0e", requery="#d62728")
    for r, L in enumerate(layers):
        ax = axes[r][0]
        for turn, _ in TURNS:
            pts = records_by_lookup.get((turn, L))
            if pts is None or len(pts) == 0:
                continue
            n = len(pts)
            x = np.linspace(0.0, 1.0, n)
            ax.plot(x, pts, color=colors.get(turn, "#333333"), lw=0.7,
                    alpha=0.8, label=turn)
        ax.axhline(1.0, color="r", ls="--", lw=0.8)
        ax.axhline(0.9, color="g", ls=":", lw=0.8)
        ax.set_ylim(0.3, 1.02)
        ax.set_ylabel(f"L{L}")
        ax.set_title(f"L{L} : projected radius vs position (ctx {size} tok)",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=4)
    axes[0][0].set_xlabel("position (fraction of context)"); 
    axes[-1][0].set_xlabel("position (fraction of context)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def disk_figure(records2d_by_turn, layers, turns, size, path,
                needle_ids_by_turn):
    """Ball-disk view: rows = layers, cols = selected turns."""
    if not HAS_MPL:
        return
    n_layers = len(layers)
    show_turns = ["doc", "requery"]
    fig, axes = plt.subplots(n_layers, len(show_turns),
                             figsize=(3.6 * len(show_turns), 3.6 * n_layers),
                             squeeze=False)
    for r, L in enumerate(layers):
        for c, turn in enumerate(show_turns):
            ax = axes[r][c]
            emb = records2d_by_turn.get((turn, L))
            if emb is None or len(emb) == 0:
                ax.set_visible(False)
                continue
            n = len(emb)
            norm = np.linspace(0.0, 1.0, n)
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=norm, s=3.5,
                            cmap="viridis", linewidths=0)
            theta = np.linspace(0, 2 * np.pi, 400)
            ax.plot(np.cos(theta), np.sin(theta), color="r", lw=1, ls="--")
            ax.plot(0.9 * np.cos(theta), 0.9 * np.sin(theta), color="g",
                    lw=0.8, ls=":")
            ax.set_aspect("equal")
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
            ax.set_title(f"L{L} · {turn} · ctx {size}", fontsize=9)
            if c == len(show_turns) - 1:
                plt.colorbar(sc, ax=ax, shrink=0.8,
                             label="pos fraction")
            for niche, mk in [("bos", "P"), ("needle", "X"), ("last", "o")]:
                i = needle_ids_by_turn.get(turn, {}).get(niche)
                if i is not None and 0 <= i < len(emb):
                    ax.scatter(emb[i, 0], emb[i, 1], color="k",
                               marker=mk, s=70, zorder=5,
                               edgecolors="w", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_figure(records, np_store, layers, size_fmt, path):
    """Cross-size summary: retrieval by turn, L15 geodesics at re-query,
    and the planted-needle radius per layer as context grows."""
    if not HAS_MPL or not records:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    sizes = sorted(set(r["size"] for r in records))
    x = np.arange(len(sizes))
    width = 0.2
    tcolors = {"doc": "#1f77b4", "query": "#2ca02c",
               "distractor": "#ff7f0e", "requery": "#d62728"}

    ax = axes[0][0]
    for t, (turn, _) in enumerate(TURNS):
        vals = []
        for s in sizes:
            rs = [r for r in records if r["size"] == s and r["turn"] == turn]
            vals.append(1.0 if rs and rs[0].get("retrieved") else 0.0)
        ax.bar(x + (t - 1.5) * width, vals, width, label=turn,
               color=tcolors[turn])
    ax.set_ylim(0, 1.15); ax.set_yticks([0, 1])
    ax.set_title("Retrieval per turn (responsiveness / forgetfulness)")
    ax.set_xticks(x); ax.set_xticklabels(size_fmt)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    vals = []
    for s in sizes:
        rs = [r for r in records if r["size"] == s and r["turn"] == "requery"]
        vals.append((rs[0]["n_context"] if rs else 0) / 1000.0)
    ax.bar(x, vals, color="#7f8c8d")
    ax.set_ylabel("K tokens")
    ax.set_title("Re-query prompt length (grows with each turn)")
    ax.set_xticks(x); ax.set_xticklabels(size_fmt)

    ax = axes[1][0]
    last = str(layers[-1])
    walk = []
    cosv = []
    for s in sizes:
        rs = [r for r in records if r["size"] == s and r["turn"] == "requery"]
        if rs and last in rs[0]["layers"]:
            e = rs[0]["layers"][last]
            walk.append(e["dP_walk"]); cosv.append(e["cos_raw_T0_Tn"])
        else:
            walk.append(float("nan")); cosv.append(float("nan"))
    ax2 = ax.twinx()
    ax.plot(x, walk, "o-", color="#d62728", label="dP_walk")
    ax2.plot(x, cosv, "s-", color="#2ca02c", label="cos(T0,Tn)")
    ax.set_xlabel("context size"); ax.set_ylabel("dP_walk")
    ax2.set_ylabel("cos")
    ax.set_xticks(x); ax.set_xticklabels(size_fmt)
    ax.set_title(f"L{last} geodesics at re-query (memory state)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")

    ax = axes[1][1]
    cm = plt.cm.viridis(np.linspace(0.15, 0.95, len(layers)))
    for i, L in enumerate(layers):
        vals = []
        for s in sizes:
            key = f"needle_radius_L{L}_{s}"
            vals.append(float(np_store.get(key, np.nan)))
        ax.plot(x, vals, "o-", color=cm[i], label=f"L{L}")
    ax.axhline(1.0, color="r", ls="--", lw=0.8)
    ax.axhline(0.9, color="g", ls=":", lw=0.8)
    ax.set_ylim(0.55, 1.02)
    ax.set_title("Planted-needle radius vs context length (depth 0.5)")
    ax.set_xlabel("context size"); ax.set_ylabel("projected radius")
    ax.set_xticks(x); ax.set_xticklabels(size_fmt)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache-dir", default="./llama32-1B-Instruct-bf16")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--sizes", default="4096,16384,32768,65536")
    ap.add_argument("--needle-depth", type=float, default=0.5)
    ap.add_argument("--needle", default=NEEDLE_DEFAULT)
    ap.add_argument("--doc-max-new-tokens", type=int, default=8)
    ap.add_argument("--ans-max-new-tokens", type=int, default=24)
    ap.add_argument("--min-answer-tokens", type=int, default=12,
                    help="force >= N tokens on answer turns so retrieval "
                         "scoring is fair (stock 1B collapses to an immediate "
                         "<|eot_id|> after long context dumps)")
    ap.add_argument("--max-fit", type=int, default=2048,
                    help="subsample size for PCA fit (64K points -> fast)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-prefix", default="logs/ball")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sizes = [int(x) for x in args.sizes.split(",")]

    os.makedirs("logs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{args.log_prefix}_{ts}"
    os.makedirs(base, exist_ok=True)

    probe = P2.DepthProbe(args.cache_dir, args.layers.split(","), device)
    layers = probe.monitored

    m = re.search(r"[-+]?\d[\d,]*", args.needle)
    needle_digits = m.group().replace(",", "") if m else None

    jsonl = open(base + ".jsonl", "w", encoding="utf-8")
    np_store = {}
    records = []
    print(f"[ball] depth={args.needle_depth} needle='{args.needle}' sizes={sizes}"
          f" layers={layers} alpha={args.alpha}")

    for size in sizes:
        doc = build_doc(probe.tokenizer, size, args.needle_depth, args.needle)
        track = P2.MetricTracker(layers, alpha=args.alpha)
        track.set_system("")
        pf_turn1 = {str(L): None for L in layers}
        records_by_lookup = {}
        records2d_by_turn = {}
        needle_ids_by_turn = {"doc": {}, "query": {}, "distractor": {},
                              "requery": {}}
        last_row = None
        for turn, user_text in TURNS:
            if turn == "doc":
                text = f"{DOC_TURN}\n\n{doc}"
                kw = dict(max_new_tokens=args.doc_max_new_tokens,
                          do_sample=False)
            else:
                text = user_text
                kw = dict(max_new_tokens=args.ans_max_new_tokens,
                          min_new_tokens=args.min_answer_tokens,
                          do_sample=False)
            reply, row = run_turn(probe, track, pf_turn1, text, kw)
            if turn == "doc" and not reply.strip():
                track.conv[-1] = {"role": "assistant", "content": "OK"}
            exact = bool(args.needle.strip() in reply)
            numeric = bool(needle_digits is not None and
                           needle_digits in reply)
            rec = dict(schema=SCHEMA, size=size, turn=turn,
                       user_text=text if turn == "doc" else user_text,
                       reply=reply, retrieved=numeric,
                       retrieved_exact=exact,
                       n_context=row["n_context"], n_gen=row["n_gen"],
                       drift_rel_prefix=row["drift_rel_prefix"],
                       layers=layer_metrics(row))
            records.append(rec)
            jsonl.write(json.dumps(json_safe(rec), ensure_ascii=False) + "\n")
            jsonl.flush()
            print(f"  [{size:>6} {turn:>10}] ctx {row['n_context']:>6} "
                  f"gen {row['n_gen']:>4} retrieved={int(numeric)} "
                  f"reply='{reply[:70]}'")

            for L in layers:
                pts = extract_points(probe, L)
                rad = pts.norm(dim=-1).cpu().float().numpy()
                emb = pca2d(pts, max_fit=args.max_fit,
                            seed=args.seed).cpu().numpy()
                records_by_lookup[(turn, L)] = rad
                records2d_by_turn[(turn, L)] = emb
                np_store[f"rad_{turn}_L{L}_{size}"] = rad.astype(np.float32)
                np_store[f"pca_{turn}_L{L}_{size}"] = emb.astype(np.float32)
                needle_ids_by_turn[turn] = dict(
                    bos=0, last=int(len(rad)) - 1,
                    needle=int(len(rad) * 0.5))
                del pts
            if turn == "doc":
                for L in layers:
                    rad = records_by_lookup[(turn, L)]
                    np_store[f"uanchor_L{L}_{size}"] = \
                        np.asarray(rad[-1], dtype=np.float32)
            torch.cuda.empty_cache()
            last_row = row
        # planted needle position within the (re-concatenated) requery prompt
        for L in layers:
            rad = records_by_lookup.get(("requery", L))
            if rad is not None and len(rad):
                needle_ids_by_turn["requery"]["needle"] = \
                    int(len(rad) * 0.5)
                np_store[f"needle_radius_L{L}_{size}"] = \
                    np.asarray(rad[needle_ids_by_turn["requery"]["needle"]],
                               dtype=np.float32)
                np_store[f"needle_radius_bos_L{L}_{size}"] = \
                    np.asarray(rad[0], dtype=np.float32)

        if not args.no_plot:
            radius_profile_figure(records_by_lookup, layers, TURNS, size,
                                  os.path.join(base, f"radius_{size}.png"),
                                  needle_ids_by_turn)
            disk_figure(records2d_by_turn, layers, TURNS, size,
                        os.path.join(base, f"disk_{size}.png"),
                        needle_ids_by_turn)
        torch.cuda.empty_cache()

    np.savez_compressed(base + "_points.npz", **np_store)
    jsonl.close()

    if not args.no_plot:
        summary_figure(records, np_store, layers, [str(s) for s in sizes],
                       base + ".png")
    print(f"\n[ball] jsonl  {base}.jsonl")
    print(f"[ball] points {base}_points.npz")
    if not args.no_plot:
        print(f"[ball] plots  {base}/index : radius_{size}.png / disk_{size}.png")
    for turn in [t for t, _ in TURNS]:
        parts = []
        for s in sizes:
            rs = [r for r in records if r["size"] == s and r["turn"] == turn]
            parts.append(f"{s}:{int(rs[0]['retrieved'])}" if rs else f"{s}:-")
        print(f"  retrieval[{turn}]: " + ", ".join(parts))


if __name__ == "__main__":
    main()