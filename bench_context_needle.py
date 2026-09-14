"""
bench_context_needle.py -- context stress-test over the Phase 2A Poincare probe.

Feeds scaling prompt lengths through the VERIFIED Phase 2A engine
(phase2_poincare_probe.py) -- the same hooks, fixed projection, Möbius
working-memory metrics -- and logs:

    * retrieval accuracy % : a needle (magic number) is buried in a filler
      document at a configurable depth; the model is asked for it and the
      answer is scored substring-exact vs the needle value.
    * hyperbolic geodesic stability : per monitored layer, d_P(T0,Tn),
      d_P(prev), cumulative walk, Möbius translation norm, cos(T0,Tn) and the
      boundary-contraction status, across the document -> query transition.
    * GPU memory footprint : torch peak-allocation (MB) under the monitored
      engine vs a hook-free stock baseline decode.

Context sizes default to the directive's 4K / 16K / 32K / 64K tokens.

Usage
-----
    python bench_context_needle.py --cache-dir llama32-1B-fp16 \
        --layers 4,8,12,16 --alpha 0.85 \
        --sizes 4096,16384,32768,65536 --needle-depth 0.5
    # quick smoke (no stock baseline, small sizes)
    python bench_context_needle.py --sizes 512,2048 --no-stock
"""
import argparse
import json
import os
import time

import torch

import phase2_poincare_probe as P2

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
QUERY_DEFAULT = ("Using only the document I just gave you, what is the magic "
                 "number? Reply with just the digits.")
SCHEMA = "bench.context_needle.v1"


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


def run_turn(probe, track, pf_turn1, user_text, gen_kwargs):
    """Stateful turn through the Phase 2A engine (mirrors the REPL loop)."""
    track.add_user(user_text)
    reply, summ = probe.run_turn(track.conv, gen_kwargs)
    track.add_assistant(reply)
    drift = {}
    layers = probe.monitored
    for L in layers:
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


def plain_generate(probe, messages, mnt):
    """Hook-free stock decode using the same model/tokenizer (baseline)."""
    inp = probe.tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=True)
    ids = torch.tensor(inp["input_ids"], device=probe.device).unsqueeze(0)
    with torch.no_grad():
        out = probe.model.generate(
            ids, use_cache=True,
            pad_token_id=int(probe.tokenizer.pad_token_id or 128004),
            max_new_tokens=mnt, do_sample=False)
    new_ids = out[0][ids.shape[1]:]
    return probe.tokenizer.decode(new_ids, skip_special_tokens=True)


def scenario(probe, document, needle_value, query, doc_mnt, ans_mnt,
             min_ans=16, alpha=0.85, engine=True):
    """Run the 2-turn doc-read/query scenario; return (success, reply2, meta).

    engine=True  -> monitored Phase 2A pipeline (dP/walk/Móbius metrics)
    engine=False -> plain generate (stock baseline; no geometric metrics)

    min_ans forces at least min_ans tokens on the answer turn so the
    retrieval test is a fair comparison even where Llama-3.2-1B's stock
    logits collapse to an immediate <|eot_id|> after a long context dump.
    """
    msg1 = ("Read the following document carefully, then respond with "
            f"exactly 'OK'.\n\n{document}")
    on_cuda = probe.device.type == "cuda"
    answer_kw = dict(max_new_tokens=ans_mnt, min_new_tokens=min_ans,
                     do_sample=False)
    if engine:
        probe.attach_monitors()
        track = P2.MetricTracker(probe.monitored, alpha=alpha)
        pf_turn1 = {str(L): None for L in probe.monitored}
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        r1, _ = run_turn(probe, track, pf_turn1, msg1,
                         dict(max_new_tokens=doc_mnt, do_sample=False))
        if not r1.strip():
            track.conv[-1] = {"role": "assistant", "content": "OK"}
        r2, row2 = run_turn(probe, track, pf_turn1, query, answer_kw)
        wall = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if on_cuda else 0.0)
        return (needle_value in r2, r2,
                dict(wall=wall, peak_mb=peak_mb, n_context=row2["n_context"],
                     n_gen=row2["n_gen"], row=row2))
    else:
        probe.detach_monitors()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        conv = [{"role": "user", "content": msg1}]
        r1 = plain_generate(probe, conv, doc_mnt)
        if not r1.strip():
            conv[-1] = {"role": "assistant", "content": "OK"}
        conv.append({"role": "user", "content": query})
        r2m = probe.tokenizer.apply_chat_template(
            conv, tokenize=True, return_dict=True, add_generation_prompt=True)
        ids = torch.tensor(r2m["input_ids"], device=probe.device).unsqueeze(0)
        with torch.no_grad():
            out = probe.model.generate(
                ids, use_cache=True,
                pad_token_id=int(probe.tokenizer.pad_token_id or 128004),
                **answer_kw)
        new_ids = out[0][ids.shape[1]:]
        r2 = probe.tokenizer.decode(new_ids, skip_special_tokens=True)
        wall = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if on_cuda else 0.0)
        return (needle_value in r2, r2,
                dict(wall=wall, peak_mb=peak_mb, n_context=None, n_gen=None,
                     row=None))


def layer_metrics(row):
    if row is None:
        return {}
    out = {}
    for k, e in row.get("layers", {}).items():
        out[str(k)] = {
            "dP_T0_Tn": e["dP_T0_Tn"],
            "dP_prev": e["dP_prev"],
            "dP_walk": e["dP_walk"],
            "mobius_norm": e["mobius_norm"],
            "cos_raw_T0_Tn": e["cos_raw_T0_Tn"],
            "prefill_radius_max": e["prefill_radius"]["max"],
            "bounded": e["prefill_radius"]["bounded"],
            "contraction_active": e["contraction"]["active"],
            "r_bos_alpha": e["contraction"]["r_bos_alpha"],
        }
    return out


def json_safe(obj):
    import math
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache-dir", default="./llama32-1B-fp16")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--sizes", default="4096,16384,32768,65536")
    ap.add_argument("--needle-depth", type=float, default=0.5)
    ap.add_argument("--needle", default=NEEDLE_DEFAULT)
    ap.add_argument("--query", default=QUERY_DEFAULT)
    ap.add_argument("--doc-max-new-tokens", type=int, default=8)
    ap.add_argument("--ans-max-new-tokens", type=int, default=24)
    ap.add_argument("--min-answer-tokens", type=int, default=16,
                    help="force >=N tokens on the answer turn so retrieval "
                         "scoring is fair even if stock logits collapse to "
                         "an immediate <|eot_id|> after a long context dump")
    ap.add_argument("--no-stock", action="store_true",
                    help="skip the hook-free stock baseline run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-prefix", default="logs/context_needle")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sizes = [int(x) for x in args.sizes.split(",")]

    probe = P2.DepthProbe(args.cache_dir, args.layers.split(","), device)
    os.makedirs("logs", exist_ok=True)

    records = []
    print(f"[needle] depth={args.needle_depth} needle='{args.needle}' "
          f"layers={probe.monitored} alpha-contraction active")
    print(f"{'size':>8} {'mode':>7} {'ok?':>4} {'peakMB':>8} "
          f"{'wall_s':>8} {'nCtx':>7} {'dPL15':>7} {'walkL15':>8} "
          f"{'cosL15':>7} {'mobL15':>7} {'bounded':>8}")
    for size in sizes:
        doc = build_doc(probe.tokenizer, size, args.needle_depth, args.needle)
        for mode, engine in (("poincare", True),
                             ("stock", False)):
            if mode == "stock" and args.no_stock:
                continue
            ok, reply, meta = scenario(probe, doc, args.needle, args.query,
                                       args.doc_max_new_tokens,
                                       args.ans_max_new_tokens,
                                       min_ans=args.min_answer_tokens,
                                       alpha=args.alpha, engine=engine)
            row = meta["row"]
            lmeta = layer_metrics(row)
            l15 = lmeta.get(str(15)) or lmeta.get(str(max(map(int, lmeta)))
                            if lmeta else "", {})
            dP = l15.get("dP_T0_Tn", float("nan"))
            walk = l15.get("dP_walk", float("nan"))
            cos_ = l15.get("cos_raw_T0_Tn", float("nan"))
            mob = l15.get("mobius_norm", float("nan"))
            bounded = l15.get("bounded", None)
            rec = dict(
                schema=SCHEMA, size=size, mode=mode,
                needle_depth=args.needle_depth, needle_value=args.needle,
                retrieved=ok, retrieval_accuracy=1.0 if ok else 0.0,
                n_context=meta["n_context"], n_gen=meta["n_gen"],
                wall_s=round(meta["wall"], 2), peak_gpu_mb=round(meta["peak_mb"], 1),
                reply=reply, layer_metrics=lmeta)
            records.append(rec)
            print(f"{size:>8} {mode:>7} {('T' if ok else 'F'):>4} "
                  f"{rec['peak_gpu_mb']:>8.1f} {rec['wall_s']:>8.2f} "
                  f"{rec['n_context'] or '':>7} {dP:>7.2f} {walk:>8.2f} "
                  f"{cos_:>7.3f} {mob:>7.3f} "
                  f"{('yes' if bounded else ('n/a' if bounded is None else 'no')):>8}")
        torch.cuda.empty_cache()

    base = args.log_prefix
    with open(base + ".jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(json_safe(r)) + "\n")
    print(f"\n[log] {base}.jsonl")
    for mode in sorted(set(r["mode"] for r in records)):
        parts = []
        for s in sizes:
            group = [r for r in records if r["size"] == s and r["mode"] == mode]
            if group:
                acc = 100.0 * sum(r["retrieval_accuracy"] for r in group) \
                    / len(group)
                parts.append(f"{s}:{acc:.0f}%")
        print(f"  retrieval acc[{mode}]: " + ", ".join(parts))


if __name__ == "__main__":
    main()