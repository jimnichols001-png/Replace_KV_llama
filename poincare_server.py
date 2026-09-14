"""
poincare_server.py -- OpenAI-compatible API over the Phase 2A Poincare probe.

A lean FastAPI/uvicorn wrapper around the VERIFIED execution engine in
phase2_poincare_probe.py (DepthProbe + MetricTracker).  No model internals,
no attention rewrites, no learned adapters -- the exact same frozen-hook +
fixed-projection pipeline Jim already validated, exposed over HTTP.

* Stateful multi-turn Poincare memory: each session owns a MetricTracker whose
  Möbius working-memory stream (u0 anchor, walk, per-layer dP/mobius trajectory)
  persists ACROSS /v1/chat/completions calls.  The model itself is re-prefilled
  from the session's message history every turn (standard transformers usage);
  the thing that persists is the Poincare state the Phase-3 engine consumes.
* Endpoints:
    GET  /v1/models            -> list {id: "llama32-1B-p2a"}
    GET  /v1/chat/completions  -> the chat page (open this URL in a browser
                                  to start a web session; / redirects here)
    POST /v1/chat/completions  -> OpenAI-compatible response (+ metrics)
* Non-standard fields (optional, on the chat request object):
    session_id       str    default "default"
    reset            bool   wipe the session's history + Poincare state
    stop_token_ids   list[int]  extra hard stop token ids (default Llama-3
                                [128001 <|end_of_text|>, 128009 <|eot_id|>])
    stop             list[str]  extra text stops (default Llama-3 eot/header
                                markers), enforced on the token stream
    include_metrics  bool   attach the per-layer dP/radius/Mó​bius row to the reply

Run:
    python poincare_server.py --cache-dir llama32-1B-Instruct-bf16 --layers 4,8,12,16 \
        --alpha 0.85 --port 8000
    Probe rows are flushed, one JSON per line, to logs/server_<timestamp>.jsonl
    (override with --log-file); stdout never carries JSON, only the banner.
"""
import argparse
import json
import os
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

import phase2_poincare_probe as P2

MODEL_ID = "llama32-1B-p2a"
DEFAULT_SAMPLE = dict(temperature=0.6, top_p=0.9, top_k=0)

# Llama-3 explicit stop configuration (see DepthProbe.run_turn for how these
# are enforced: eos_token_id hard stops + token-stream clamping).
LLAMA3_STOP_TOKEN_IDS = [128001, 128009]      # <|end_of_text|>, <|eot_id|>
LLAMA3_STOP_STRINGS = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]

# the in-browser chat page (GET / and GET /v1/chat/completions); POST on
# /v1/chat/completions stays the OpenAI-compatible JSON API.
CHAT_PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "web_chat.html"), encoding="utf-8").read()

app = FastAPI(title="Poincare KV Memory Server (Phase 2A)",
              version="0.1.0", docs_url="/docs")

# --------------------------------------------------------------------------
# server state (populated at startup)
# --------------------------------------------------------------------------
S = {}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    session_id: str = "default"
    reset: bool = False
    include_metrics: bool = False
    stop_token_ids: list[int] | None = None
    stop: list[str] | None = None


# --------------------------------------------------------------------------
# stateful session: the Poincare working-memory stream
# --------------------------------------------------------------------------
class Session:
    def __init__(self, system: str):
        self.tr = P2.MetricTracker(S["layers"], alpha=S["alpha"])
        if system:
            self.tr.set_system(system)
        self.pf_turn1 = {str(L): None for L in S["layers"]}

    def reset(self, system: str = None):
        self.tr = P2.MetricTracker(S["layers"], alpha=S["alpha"])
        if system:
            self.tr.set_system(system)
        self.pf_turn1 = {str(L): None for L in S["layers"]}


def run_turn(sess: Session, user_text: str, gen_kwargs: dict):
    """One stateful conversational turn through the Phase 2A engine."""
    tr = sess.tr
    tr.add_user(user_text)
    reply, summ = S["probe"].run_turn(tr.conv, gen_kwargs)
    tr.add_assistant(reply)
    drift = {}
    for L in S["layers"]:
        key = str(L)
        if sess.pf_turn1[key] is not None:
            cur = summ[key]["pf_full"]
            ref = sess.pf_turn1[key]
            Ln = ref.shape[0]
            ref_n = ref.norm(dim=-1).clamp_min(1e-6)
            delta = (cur[:Ln] - ref).abs()
            rel = (delta / ref_n.unsqueeze(-1)).max(dim=-1).values
            drift[L] = (float(rel.max().item()), float(delta[-1].max().item()))
        else:
            sess.pf_turn1[key] = summ[key]["pf_full"]
    row = tr.record(user_text, reply, summ, drift=drift)
    return reply, row


def json_safe(obj):
    """Recursively replace non-finite floats (NaN/Inf, e.g. turn-1 inflation)
    with None so the strict JSON encoders accept the metrics row."""
    if isinstance(obj, float):
        import math
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def log_row(row, body):
    """Append one turn's probe row to the server's .jsonl (best effort --
    logging failures must never break the API response).  Probe JSON lives in
    this file exclusively; stdout carries nothing but the response payload."""
    f = S.get("jsonl")
    if f is None or row is None:
        return
    try:
        rec = {"ts": int(time.time()), "session_id": body.session_id,
               "model": body.model or MODEL_ID}
        rec.update(json_safe(row))
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
    except Exception:
        pass


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "phase2a",
            "meta": {"layers": S["layers"], "alpha": S["alpha"]},
        }],
    }


@app.get("/v1/chat/completions", response_class=HTMLResponse)
def chat_page():
    """Browser chat session: GET renders the UI, POST runs a turn."""
    return CHAT_PAGE.replace("{{MODEL_ID}}", MODEL_ID)


@app.get("/", response_class=HTMLResponse)
def index():
    return RedirectResponse(url="/v1/chat/completions")


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest):
    if body.reset or body.session_id not in S["sessions"]:
        S["sessions"][body.session_id] = Session(S["system"])
    sess = S["sessions"][body.session_id]

    if body.max_tokens:
        mnt = body.max_tokens
    else:
        mnt = S["default_max_new_tokens"]
    gen_kwargs = dict(max_new_tokens=mnt, do_sample=True,
                      temperature=body.temperature if body.temperature
                      is not None else DEFAULT_SAMPLE["temperature"],
                      top_p=body.top_p if body.top_p is not None
                      else DEFAULT_SAMPLE["top_p"],
                      top_k=body.top_k if body.top_k is not None
                      else DEFAULT_SAMPLE["top_k"],
                      stop_token_ids=body.stop_token_ids
                      or LLAMA3_STOP_TOKEN_IDS,
                      stop_strings=body.stop or LLAMA3_STOP_STRINGS)

    last_reply = ""
    last_row = None
    for msg in body.messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if role == "assistant":
            sess.tr.add_assistant(content)
            continue
        if role == "system" and not sess.tr.conv:
            sess.tr.set_system(content)
            continue
        last_reply, last_row = run_turn(sess, content, gen_kwargs)
        log_row(last_row, body)

    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": last_reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": last_row["n_context"] if last_row else 0,
            "completion_tokens": last_row["n_gen"] if last_row else 0,
            "total_tokens": (last_row["n_context"] + last_row["n_gen"])
            if last_row else 0,
        },
    }
    if body.include_metrics and last_row is not None:
        resp["metrics"] = json_safe(last_row)
    return resp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache-dir", default="./llama32-1B-Instruct-bf16")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--system", default="")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-file", default=None,
                    help="jsonl file to append probe rows "
                         "(default logs/server_<timestamp>.jsonl)")
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    probe = P2.DepthProbe(args.cache_dir, args.layers.split(","), device)
    S["probe"] = probe
    S["layers"] = probe.monitored
    S["alpha"] = args.alpha
    S["system"] = args.system
    S["default_max_new_tokens"] = args.max_new_tokens
    S["sessions"] = {}
    os.makedirs("logs", exist_ok=True)
    log_path = args.log_file or \
        f"logs/server_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    S["jsonl"] = open(log_path, "a", encoding="utf-8")
    print(f"[server] probe rows -> {log_path}")
    print(f"[server] {MODEL_ID} | layers {S['layers']} | alpha {S['alpha']} "
          f"| stop_ids {LLAMA3_STOP_TOKEN_IDS} | stop {LLAMA3_STOP_STRINGS} "
          f"| docs at /docs | http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()