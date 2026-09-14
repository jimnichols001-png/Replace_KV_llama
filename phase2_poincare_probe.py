"""
phase2_poincare_probe.py -- Poincaire control probe on a STOCK, frozen
Llama-3.2-1B-Instruct.  Phase 2A: multi-layer depth probing.

Phase 2A extends Phase 2 from a single layer-8 monitor to simultaneous depth
checkpoints and adds a boundary-contraction test.  Still zero weights modified:
the model is fp16, eval(), fully frozen.

Directives implemented
----------------------
  1. Multi-layer hooks            -- capture hidden states at layers 4, 8, 12
                                     and the final layer (index 15, the "16th"
                                     output) of the 16-layer stack, all at once.
  2. Multi-layer trajectory       -- JSONL + PNG now log, per monitored layer:
        r_L     projected-radius bounds (min/max/mean/std), bounded<1 check
        d_P(T0,Tn)_L, d_P(prev)_L, dP_walk_L, cos_raw(T0,Tn)_L
        Mobius translation vector  t = (-u_{n-1}) (+) u_n   (norm + vector)
        alpha-contracted stream   dP_T0_Tn_alpha_L, dP_walk_alpha_L
  3. Boundary contraction check   -- unlearned interior scaling
        h_alpha = alpha * proj(h),  alpha ~ 0.85
        applied to the high-norm BOS token (radius 0.9987 -> ~0.849) to pull
        the boundary shell into the active volume of B^2048.  Verified for
        EVERY monitored layer (r_bos_raw/r_bos_alpha/margin/active).

Geometry (fixed, no learned W_mem)
----------------------------------
        proj(h) = h / (1 + sqrt(1 + ||h||^2))          R^2048 -> open ball
        d_P(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
        u (+) v = ((1+2<u,v>+||v||^2)u + (1-||u||^2)v) / (1+2<u,v>+||u||^2||v||^2)

Snapshots per monitored layer L, per turn:
        u_n^L  prompt-side : proj(h_L of FINAL prompt token)  (context summary
                             the model reads BEFORE replying)
        a_n^L  reply-side  : proj(h_L of FINAL generated token) (memory AFTER)
   T0 = u_1^L.

Causal pipeline self-test (per layer): every token of the turn-1 prompt must be
reproduced identically in later prefills (attention is causal).  Reported as a
RELATIVE per-position error (|delta|/||ref||) so the boundary-pinned BOS
activation (norm ~800) cannot mask small genuine differences.

Output
------
* layers 4/8/12/15 per-turn table on stdout (gateway metrics per layer).
* logs/<base>.jsonl     : one record per turn; per-layer dict under "layers".
* logs/<base>.meta.json : run header (incl. layers + alpha) + q1/q2/q3 summary.
* logs/<base>.png       : 2x2 multi-layer trajectory + contraction plot.

Usage
-----
    # interactive (classic control test, now 4 depth checkpoints)
    python phase2_poincare_probe.py --cache-dir llama32-1B-fp16 \
        --layers 4,8,12,16 --alpha 0.85 \
        --temperature 0.6 --top-p 0.9 --max-new-tokens 64

    # deterministic baseline stream (greedy) -- what Phase 3 diffs against
    python phase2_poincare_probe.py --cache-dir llama32-1B-fp16 \
        --script tests/demo_turns.json --max-new-tokens 64

REPL meta-commands:  :reset   (clear history, restart the trajectory)
                     :metrics (re-print the current metric table)
                     :quit    (exit)
"""
import argparse
import json
import os
import sys
import time
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

SCHEMA = "phase2.probe.v2"
ACTIVE_R = 0.9          # contracted radii <= this count as "active volume"


# ============================================================================
# 0. torch-2.5.x SDPA GQA regression workaround.
# On this Windows build, SDPA with `enable_gqa=True` silently falls back to the
# fp32 MATH backend during PREFILL (q_len > 1): it materializes
# [n_heads, seq, seq] float32 attention (~32 GiB at 16K context), OOMing on
# any context beyond ~5K tokens.  Decode (q_len=1) is fine -- only prefill.
# Forcing GQA-in-SDPA OFF routes attention through the repeat-kv expansion
# (KV heads 8 -> 32) and the memory-bounded flash/mem_efficient kernels, which
# are the path this repo measured (bounded ~1/2 GB even at 64K context).
# ============================================================================
def _disable_gqa_in_sdpa():
    try:
        import transformers.integrations.sdpa_attention as _sdpa
        _sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
    except Exception:
        pass


_disable_gqa_in_sdpa()


# ============================================================================
# 1. THE FIXED GEOMETRY  (the entire "metric injection" -- no weights trained)
# ============================================================================
def project(h: torch.Tensor) -> torch.Tensor:
    """h [..., d] -> open unit ball via  h / (1 + sqrt(1 + ||h||^2))."""
    n = h.norm(dim=-1, keepdim=True)
    return h / (1.0 + (1.0 + n * n).sqrt())


def poincare_dist(u: torch.Tensor, v: torch.Tensor) -> float:
    """d_P(u, v) in the Poincare ball.  u, v [d]."""
    du = max(1.0 - float(u.double().dot(u.double())), 1e-6)
    dv = max(1.0 - float(v.double().dot(v.double())), 1e-6)
    num = 2.0 * float((u.double() - v.double()).norm().item() ** 2)
    arg = max(1.0 + num / (du * dv), 1.0 + 1e-6)
    return float(torch.acosh(torch.tensor(arg, dtype=torch.float64)).item())


def mobius_translate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mobius gyrovector that translates a -> b:  t = (-a) (+) b.

    So t lands at 0 when a == b (turn-1 idle), and its norm grows sharply as
    either endpoint approaches the boundary -- it is the geodesic "velocity"
    of the working memory between successive turn snapshots."""
    a = a.double()
    b = b.double()
    na = -a
    aa = float((a * a).sum().item())
    bb = float((b * b).sum().item())
    ab = float((b * na).sum().item())
    denom = max(1.0 + 2.0 * ab + aa * bb, 1e-9)
    num = (1.0 + 2.0 * ab + bb) * na + (1.0 - aa) * b
    return num / denom


# ============================================================================
# 2. THE PROBE   (loaded stock + frozen, multi-layer monitor hooks)
# ============================================================================
class DepthProbe:
    def __init__(self, cache_dir: str, monitored, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        t0 = time.time()
        print(f"[load] {cache_dir}  (fp16, frozen, layer hooks "
              f"{list(monitored)})")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cache_dir, dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                cache_dir, torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        stack = self.model.model.layers
        self.n_layers = len(stack)
        self.monitored = []
        for L in monitored:
            L = int(L)
            if L < 0 or L >= self.n_layers:
                print(f"[probe] clamping monitored layer {L} -> "
                      f"{self.n_layers - 1} (16-layer stack)")
                L = self.n_layers - 1
            if L not in self.monitored:
                self.monitored.append(L)
        self.captured = {L: [] for L in self.monitored}
        self.handles = []
        self._attach_monitors()
        print(f"[load] {time.time()-t0:.1f}s  "
              f"{sum(p.numel() for p in self.model.parameters())/1e9:.3f}B  "
              f"hidden={self.model.config.hidden_size}  "
              f"monitored={self.monitored}")

    def _hook(self, L):
        def hook(_m, _a, out, _L=L):
            t = out[0] if isinstance(out, tuple) else out
            self.captured[_L].append(t.detach())
        return hook

    def _attach_monitors(self):
        """Register the non-blocking layer-output hooks (idempotent)."""
        self.captured = {L: [] for L in self.monitored}
        self.handles = []
        for L in self.monitored:
            self.handles.append(
                self.model.model.layers[L].register_forward_hook(self._hook(L)))

    def detach_monitors(self):
        """Remove the hooks (hook-free = true stock forward, clone of the
        baseline decode for GPU-memory measurement)."""
        for h in self.handles:
            h.remove()
        self.handles = []
        self.captured = {}

    def attach_monitors(self):
        if not self.handles:
            self._attach_monitors()

    # ------------------------------------------------------------ per turn
    def run_turn(self, messages, gen_kwargs):
        """Render the Llama-3 chat template, generate with explicit Llama-3
        stop tokens, and return (reply_text, {str(L): layer_summary}).

        gen_kwargs may carry (both optional):
            stop_token_ids  extra token ids that end generation (default
                            Llama-3: [128001 <|end_of_text|>,
                            128009 <|eot_id|>])
            stop_strings    extra text markers that terminate the reply
                            (default Llama-3: ["<|eot_id|>",
                            "<|end_of_text|>", "<|start_header_id|>"])

        Every other key is forwarded verbatim to model.generate(); the
        caller's dict is never mutated.  The generated token stream is
        clamped at the earliest COMPLETE stop marker, so a partial marker at
        the max-length boundary is also dropped and no raw header/template
        tokens ever leak into the returned text."""
        for k in self.captured:
            self.captured[k].clear()

        kw = dict(gen_kwargs)                      # never touch the caller
        stop_token_ids = [int(x) for x in kw.pop("stop_token_ids", [])]
        if not stop_token_ids:
            stop_token_ids = [128001, 128009]      # <|end_of_text|>, <|eot_id|>
        stop_strings = list(kw.pop("stop_strings", []))
        if not stop_strings:
            stop_strings = ["<|eot_id|>", "<|end_of_text|>",
                            "<|start_header_id|>"]

        # Stock Llama-3.2-Instruct chat template rendered to *text* first, then
        # tokenized, so the system/user/assistant framing matches the official
        # template exactly (add_generation_prompt=True appends the assistant
        # header so generation begins cleanly).
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        if enc.get("attention_mask") is not None:
            kw["attention_mask"] = enc["attention_mask"].to(self.device)

        # Hard stop-token ids (Llama-3) -- union of our defaults and any the
        # caller explicitly set -- so the decoder halts at <|eot_id|> and can
        # never roll into a new <|start_header_id|> header.
        user_eos = kw.pop("eos_token_id", None)
        merged_eos = list(stop_token_ids)
        if user_eos is not None:
            if isinstance(user_eos, int):
                user_eos = [user_eos]
            merged_eos.extend(int(x) for x in user_eos)
        kw["eos_token_id"] = list(dict.fromkeys(merged_eos))
        kw.setdefault("pad_token_id",
                      int(self.tokenizer.pad_token_id or 128004))

        with torch.no_grad():
            out = self.model.generate(ids, use_cache=True, **kw)
        new_ids = out[0][ids.shape[1]:]                    # generated tokens
        new_ids = self._clamp_at_stop(new_ids, stop_strings)
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        summary = self._summarise_capture(ids.shape[1], len(new_ids))
        return reply, summary

    def _clamp_at_stop(self, ids, stop_strings):
        """Truncate generated token-ids at the earliest complete stop marker.

        Matching is done on the raw token stream (not on decoded text), so a
        marker split by the max-length boundary is also removed.  Returns ids
        unchanged when nothing matches."""
        if ids.dim() == 2:
            ids = ids[0]
        n = ids.shape[0]
        cut = n
        for marker in stop_strings:
            mt = self.tokenizer.encode(marker, add_special_tokens=False)
            k = len(mt)
            if not k or k > n:
                continue
            t = torch.as_tensor(mt, device=ids.device)
            for s in range(n - k + 1):
                if torch.equal(ids[s:s + k], t):
                    cut = min(cut, s)
                    break
        return ids[:cut] if cut < n else ids

    def _summarise_capture(self, n_prompt, n_gen):
        """Turn hook brams into per-layer snapshots + radius stats."""
        out = {}
        for L in self.monitored:
            chains = self.captured[L]
            prefill = [c for c in chains if c.shape[1] > 1]
            decode = [c for c in chains if c.shape[1] == 1]
            if prefill:
                pf = prefill[0].float()[0]                  # [P, 2048]
            else:
                pf = torch.cat(decode, dim=1).float()[0]
            p_proj = project(pf)
            radii = p_proj.norm(dim=-1)
            u_last_raw = pf[-1]
            u_last_proj = project(u_last_raw)
            a_last_raw = (decode[-1][0, 0].float() if decode else u_last_raw)
            a_last_proj = project(a_last_raw)
            out[str(L)] = dict(
                pf_full=self._strip(pf),
                u_last_raw=self._strip(u_last_raw),
                u_last_proj=self._strip(u_last_proj),
                a_last_raw=self._strip(a_last_raw),
                a_last_proj=self._strip(a_last_proj),
                r_bos=float(radii[0].item()),
                radius_stats=dict(
                    min=float(radii.min().item()),
                    max=float(radii.max().item()),
                    mean=float(radii.mean().item()),
                    std=float(radii.std().item()),
                    bounded=bool(float(radii.max().item()) < 1.0),
                    n=len(radii)),
                n_prompt=n_prompt, n_gen=n_gen)
        return out

    @staticmethod
    def _strip(t):
        return t.cpu().float()


# ============================================================================
# 3. TRACKER -- accumulates the per-layer T0-anchored metric stream
# ============================================================================
class MetricTracker:
    def __init__(self, layers, alpha: float, save_vectors: bool = False):
        self.layers = [int(L) for L in layers]
        self.alpha = alpha
        self.save_vectors = save_vectors
        self.u0_proj = {L: None for L in self.layers}
        self.u0_raw = {L: None for L in self.layers}
        self._prev_u = {L: None for L in self.layers}
        self.walk = {L: 0.0 for L in self.layers}
        self.walk_alpha = {L: 0.0 for L in self.layers}
        self.rows = []
        self.conv = []

    def set_system(self, system: str):
        if system:
            self.conv.insert(0, {"role": "system", "content": system})

    def add_user(self, text):
        self.conv.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.conv.append({"role": "assistant", "content": text})

    # ------------------------------------------------------------- record
    def record(self, text_user, text_reply, s: dict,
               drift: dict = None) -> dict:
        """
        drift: {int L: (rel_prefix, anchor_abs)} per monitored layer, from the
        causal prefill self-test (computed in the run loop).
        """
        turn = len(self.rows) + 1
        drift = drift or {}
        layers = {}
        max_rel = None
        max_anchor = None
        for L in self.layers:
            sL = s[str(L)]
            u = sL["u_last_proj"]
            a = sL["a_last_proj"]
            if self.u0_proj[L] is None:
                self.u0_proj[L] = u
                self.u0_raw[L] = sL["u_last_raw"]
            prev = self._prev_u[L] if self._prev_u[L] is not None \
                else self.u0_proj[L]

            d_t0 = poincare_dist(self.u0_proj[L], u)
            d_prev = poincare_dist(prev, u)
            self.walk[L] += d_prev
            self._prev_u[L] = u

            a_p = self.alpha
            u0_a = a_p * self.u0_proj[L]
            u_a = a_p * u
            prev_a = a_p * prev
            d_t0_a = poincare_dist(u0_a, u_a)
            d_prev_a = poincare_dist(prev_a, u_a)
            self.walk_alpha[L] += d_prev_a

            cos0 = torch.nn.functional.cosine_similarity(
                self.u0_raw[L].unsqueeze(0),
                sL["u_last_raw"].unsqueeze(0)).item()
            euclid = float((u - self.u0_proj[L]).norm().item())
            euclid_prev = float((u - prev).norm().item())

            mob = mobius_translate(prev, u)
            mob_norm = float(mob.norm().item())

            rs = sL["radius_stats"]
            r_max_a = a_p * rs["max"]
            r_bos_a = a_p * sL["r_bos"]
            entry = dict(
                prefill_radius=rs,
                contraction=dict(
                    alpha=a_p,
                    r_max_raw=rs["max"],
                    r_max_alpha=r_max_a,
                    r_bos_raw=sL["r_bos"],
                    r_bos_alpha=r_bos_a,
                    margin_from_boundary=1.0 - r_max_a,
                    active=bool(r_max_a <= ACTIVE_R)),
                u_radius=float(u.norm().item()),
                u_raw_norm=float(sL["u_last_raw"].norm().item()),
                a_radius=float(a.norm().item()),
                dP_T0_Tn=d_t0,
                dP_prev=d_prev,
                dP_walk=self.walk[L],
                dP_T0_Tn_alpha=d_t0_a,
                dP_walk_alpha=self.walk_alpha[L],
                cos_raw_T0_Tn=float(cos0),
                euclid_proj_T0_Tn=euclid,
                inflation=d_prev / (euclid_prev + 1e-6)
                if euclid_prev > 1e-9 else float("nan"),
                mobius_norm=mob_norm,
            )
            rel, anchor = drift.get(L, (None, None))
            if rel is not None:
                max_rel = rel if max_rel is None else max(max_rel, rel)
                max_anchor = anchor if max_anchor is None \
                    else max(max_anchor, anchor)
            entry["drift_rel_prefix"] = rel
            entry["drift_anchor_abs"] = anchor
            if self.save_vectors:
                entry["u_proj"] = u.tolist()
                entry["a_proj"] = a.tolist()
                entry["mobius_vec"] = mob.tolist()
            layers[str(L)] = entry

        row = dict(schema=SCHEMA, turn=turn, user_text=text_user,
                   reply=text_reply, n_context=s[str(self.layers[0])]
                   ["n_prompt"], n_gen=s[str(self.layers[0])]["n_gen"],
                   drift_rel_prefix=max_rel, drift_anchor_abs=max_anchor,
                   layers=layers)
        self.rows.append(row)
        return row

    # -------------------------------------------------------------- table
    def table(self):
        cols = [("trn", 3), ("ctxT", 5), ("genT", 5)]
        width = {c: w for c, w in cols}
        header = "".join(f"{c:>{w}}" for c, w in cols)
        for L in self.layers:
            header += f"{('dP.L'+str(L)):>10}"
            header += f"{('u.L'+str(L)):>9}"
        lines = [header]
        for r in self.rows:
            line = (f"{r['turn']:>3} {r['n_context']:>5} {r['n_gen']:>5}")
            for L in self.layers:
                e = r["layers"][str(L)]
                line += f"{e['dP_T0_Tn']:>10.3f}"
                line += f"{e['u_radius']:>9.4f}"
            lines.append(line)
        return "\n".join(lines)

    # ---------------------------------------------------------------- meta
    def meta(self, header: dict) -> dict:
        last = self.rows[-1] if self.rows else None
        per_layer = {}
        if last:
            for L in self.layers:
                e = last["layers"][str(L)]
                per_layer[str(L)] = dict(
                    dP_T0_Tn=e["dP_T0_Tn"], dP_walk=e["dP_walk"],
                    u_radius=e["u_radius"], a_radius=e["a_radius"],
                    prefill_max_radius=e["prefill_radius"]["max"],
                    bounded=e["prefill_radius"]["bounded"],
                    contraction=e["contraction"],
                    cos_raw_T0_Tn=e["cos_raw_T0_Tn"])
        return dict(
            header=header,
            n_turns=len(self.rows),
            final=per_layer,
            q1_native_manifold_drift=dict(
                layers=self.layers,
                observation="BOS pos-0 norm ~800 pins to the boundary "
                            "shell (r~0.999) at every monitored depth while "
                            "content tokens sit at r~0.78-0.87; alpha=%.2f "
                            "contraction pulls the shell into the active "
                            "volume (r<=%.1f)." % (self.alpha, ACTIVE_R)),
            q2_working_memory_trajectory=dict(
                layers=self.layers,
                dP_T0_Tn_series={str(L): [r["layers"][str(L)]["dP_T0_Tn"]
                                          for r in self.rows]
                                 for L in self.layers},
                dP_walk_series={str(L): [r["layers"][str(L)]["dP_walk"]
                                         for r in self.rows]
                                for L in self.layers},
                dP_T0_Tn_alpha_series={
                    str(L): [r["layers"][str(L)]["dP_T0_Tn_alpha"]
                             for r in self.rows]
                    for L in self.layers}),
            q3_ablation_baseline=dict(
                protocol="greedy deterministic + scripted turns -> this JSONL",
                script_feeds=header.get("script"),
                schema=SCHEMA,
                layers=self.layers,
                alpha=self.alpha,
                diff_keys=["dP_T0_Tn", "dP_walk", "cos_raw_T0_Tn",
                           "prefill_radius.max", "mobius_norm"],
            ),
        )


# ============================================================================
# 4. PLOT  (2x2: per-layer dP / radius / walk / contraction check)
# ============================================================================
def plot_trajectory(rows, layers, path):
    if not HAS_MPL:
        return
    if not rows:
        return
    turns = [r["turn"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    cm = plt.cm.viridis([0.15, 0.45, 0.7, 0.95])

    ax = axes[0, 0]
    for i, L in enumerate(layers):
        ax.plot(turns, [r["layers"][str(L)]["dP_T0_Tn"] for r in rows],
                "o-", color=cm[i], label=f"layer {L}")
    ax.set_title("d_P(T0, Tn) per monitored layer")
    ax.set_xlabel("turn"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, L in enumerate(layers):
        ax.plot(turns, [r["layers"][str(L)]["u_radius"] for r in rows],
                "o-", color=cm[i], label=f"u layer {L}")
        ax.plot(turns, [r["layers"][str(L)]["prefill_radius"]["max"]
                        for r in rows], "s--", color=cm[i], lw=0.8)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="boundary r=1")
    ax.axhline(ACTIVE_R, color="g", ls=":", lw=0.8, label=f"active r<={ACTIVE_R}")
    ax.set_title("Projected radius: u_n (solid) / prefill max (dashed)")
    ax.set_xlabel("turn"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i, L in enumerate(layers):
        ax.plot(turns, [r["layers"][str(L)]["dP_walk"] for r in rows],
                "o-", color=cm[i], label=f"layer {L}")
        ax.plot(turns, [r["layers"][str(L)]["dP_walk_alpha"] for r in rows],
                "--", color=cm[i], lw=0.8)
    ax.set_title("Cumulative working-memory walk (solid) / alpha (dashed)")
    ax.set_xlabel("turn"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    last = rows[-1]
    keys = [str(L) for L in layers]
    raw = [last["layers"][k]["contraction"]["r_bos_raw"] for k in keys]
    con = [last["layers"][k]["contraction"]["r_bos_alpha"] for k in keys]
    x = range(len(layers))
    ax.bar([i - 0.18 for i in x], raw, 0.36, label=f"raw (pinned ~1.0)")
    ax.bar([i + 0.18 for i in x], con, 0.36,
           label=f"alpha={last['layers'][keys[0]]['contraction']['alpha']}")
    ax.axhline(ACTIVE_R, color="g", ls=":", lw=1, label=f"active r<={ACTIVE_R}")
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="boundary")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"L{k}" for k in keys])
    ax.set_ylim(0, 1.05)
    ax.set_title("Boundary contraction: BOS radius raw vs alpha-scaled")
    ax.legend(fontsize=8)

    fig.suptitle(f"Phase 2A stock control probe -- {len(layers)} monitored "
                 f"layers {keys}, {len(rows)} turns", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================
# 5. RUN LOOP (REPL or --script)
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache-dir", default="./llama32-1B-fp16")
    ap.add_argument("--layer", type=int, default=None,
                    help="single monitored layer (overrides --layers)")
    ap.add_argument("--layers", default="4,8,12,16",
                    help="comma list of monitored depth checkpoints; values "
                         "> n_layers clamp to the final layer (16 -> 15)")
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="unlearned interior contraction scale (h -> alpha*h)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--system", default="",
                    help="optional system prompt (default: stock template)")
    ap.add_argument("--script", default=None,
                    help="JSON {'system':..., 'turns':[...]} greedy replay")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--save-vectors", action="store_true",
                    help="keep per-layer u/a/mobius vectors in the JSONL "
                         "(adds ~4*12 KB/turn)")
    ap.add_argument("--log-prefix", default=None,
                    help="output base for .jsonl/.meta.json/.png")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_script(path):
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data.get("turns"), list), \
        "script file needs a 'turns' list of user strings"
    return data.get("system", ""), data["turns"]


def main():
    args = parse_args()
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("logs", exist_ok=True)
    base = args.log_prefix or \
        f"logs/phase2_{time.strftime('%Y%m%d_%H%M%S')}"
    probe = DepthProbe(args.cache_dir, args.layers.split(","), device)
    monitored = probe.monitored
    if args.layer is not None:
        if args.layer >= probe.n_layers:
            args.layer = probe.n_layers - 1
        monitored = [args.layer]
        print(f"[probe] --layer override -> monitored = {monitored}")

    tr = MetricTracker(monitored, alpha=args.alpha,
                       save_vectors=args.save_vectors)
    tr.set_system(args.system)
    pf_turn1 = {str(L): None for L in monitored}

    script_system = ""
    script_turns = []
    if args.script:
        script_system, script_turns = load_script(args.script)
        tr.set_system(script_system)
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
        src_desc = f"--script {args.script}"
    else:
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                          temperature=args.temperature, top_p=args.top_p,
                          top_k=args.top_k)
        src_desc = "interactive REPL"

    header = dict(schema=SCHEMA, model=os.path.abspath(args.cache_dir),
                  layers=monitored, alpha=args.alpha, seed=args.seed,
                  sampling=src_desc, script=args.script or None,
                  system=args.system or script_system,
                  max_new_tokens=args.max_new_tokens,
                  do_sample=not args.script)
    print(f"[probe] {src_desc} | layers {monitored} | alpha {args.alpha} | "
          f"seed {args.seed} | gen: {gen_kwargs}")
    print("type :reset :metrics :quit inside the REPL\n")

    def process_turn(user_text):
        nonlocal pf_turn1
        tr.add_user(user_text)
        reply, summ = probe.run_turn(tr.conv, gen_kwargs)
        tr.add_assistant(reply)
        drift = {}
        for L in monitored:
            key = str(L)
            if pf_turn1[key] is not None:
                cur = summ[key]["pf_full"]
                ref = pf_turn1[key]
                L_n = ref.shape[0]
                ref_n = ref.norm(dim=-1).clamp_min(1e-6)     # [L]
                delta = (cur[:L_n] - ref).abs()              # [L, d]
                rel = (delta / ref_n.unsqueeze(-1)).max(dim=-1).values
                drift[L] = (float(rel.max().item()),
                            float(delta[-1].max().item()))
            else:
                pf_turn1[key] = summ[key]["pf_full"]
        row = tr.record(user_text, reply, summ, drift=drift)
        return reply, row

    def print_row(r):
        dr = r["drift_rel_prefix"]
        da = r["drift_anchor_abs"]
        dstr = f"{dr:.1e}" if dr is not None else "  -  "
        astr = f"{da:.1e}" if da is not None else "  -  "
        print(f"  turn {r['turn']}: ctx {r['n_context']}t gen {r['n_gen']}t "
              f"| driftRel(max)={dstr} anchorAbs={astr}")
        for L in monitored:
            e = r["layers"][str(L)]
            c = e["contraction"]
            dd = e["drift_rel_prefix"]
            ddstr = f"{dd:.1e}" if dd is not None else "   -  "
            print(f"    L{str(L):<2} u={e['u_radius']:.4f} "
                  f"a={e['a_radius']:.4f} rx=[{e['prefill_radius']['min']:.4f},"
                  f"{e['prefill_radius']['max']:.4f}] "
                  f"dP(T0,Tn)={e['dP_T0_Tn']:.3f} dp_={e['dP_prev']:.3f} "
                  f"walk={e['dP_walk']:.3f} cos0={e['cos_raw_T0_Tn']:.3f} "
                  f"mobius={e['mobius_norm']:.4f} driftRel={ddstr}"
                  f"  | C:BOS {c['r_bos_raw']:.4f}->{c['r_bos_alpha']:.4f} "
                  f"active={int(c['active'])}")

    def write_run():
        with open(base + ".jsonl", "w") as f:
            for r in tr.rows:
                f.write(json.dumps(r) + "\n")
        with open(base + ".meta.json", "w") as f:
            json.dump(tr.meta(header), f, indent=2)
        plot_trajectory(tr.rows, monitored, base + ".png")

    def finish():
        print("\n" + tr.table())
        write_run()
        print(f"\n[log] {base}.jsonl  | {base}.meta.json  | "
              f"{base}.png (kwargs r-save={args.save_vectors})")

    if args.script:
        turns = script_turns[:args.max_turns] if args.max_turns \
            else script_turns
        for i, u in enumerate(turns, 1):
            reply, row = process_turn(u)
            print_row(row)
        finish()
        return

    # ---- interactive REPL
    print("... bootstrapping T0.  Say something (or :quit)")
    try:
        while True:
            try:
                line = input("\n» ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in (":quit", ":q"):
                break
            if line in (":reset",):
                tr.rows.clear()
                tr.walk = {L: 0.0 for L in monitored}
                tr.walk_alpha = {L: 0.0 for L in monitored}
                tr.u0_proj = {L: None for L in monitored}
                tr.u0_raw = {L: None for L in monitored}
                tr._prev_u = {L: None for L in monitored}
                tr.conv = []
                pf_turn1 = {str(L): None for L in monitored}
                if args.system:
                    tr.set_system(args.system)
                print("  (history cleared)")
                continue
            if line in (":metrics", ":m"):
                print(tr.table())
                continue
            reply, row = process_turn(line)
            print_row(row)
            if args.max_turns and len(tr.rows) >= args.max_turns:
                break
    except KeyboardInterrupt:
        print("\n  (interrupted)")
    finish()


if __name__ == "__main__":
    main()