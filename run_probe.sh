#!/usr/bin/env bash
#
# run_probe.sh -- single-entry runner for phase2_poincare_probe.py.
# Automatically resolves the sandbox_env Python binary and manages log
# prefixes, so no long CLI commands are needed.
#
# Usage:
#   ./run_probe.sh repl                          interactive REPL (Phase 2A)
#   ./run_probe.sh baseline                      scripted greedy baseline -> logs/
#   ./run_probe.sh vectors                       quick --save-vectors debug dump
#   ./run_probe.sh repl --max-new-tokens 128     extra flags are passed through
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- resolve the sandbox_env python ---------------------------------------
PY=""
for c in \
    "$ROOT/../miniconda3/envs/sandbox_env/bin/python" \
    "/home/bedroom_pc/miniconda3/envs/sandbox_env/bin/python" \
    "$HOME/miniconda3/envs/sandbox_env/bin/python"; do
    if [[ -x "$c" ]]; then PY="$c"; break; fi
done
if [[ -z "$PY" ]]; then
    PY="$(command -v python 2>/dev/null || true)"
    echo "[run_probe] sandbox_env python not found; falling back to: $PY" >&2
fi
if [[ -z "$PY" ]]; then
    echo "[run_probe] no python found -- activate the sandbox_env conda env first." >&2
    exit 1
fi

CACHE="$ROOT/llama32-1B-Instruct-bf16"
SCRIPT="$ROOT/tests/demo_turns.json"
PROBE="$ROOT/phase2_poincare_probe.py"

usage() {
    cat <<EOF
Usage: $0 {repl|baseline|vectors} [extra --flags]

  repl       interactive REPL: layers 4,8,12,16, alpha 0.85, temp 0.6
             logs -> logs/phase2a_repl.*
  baseline   deterministic scripted re-run over tests/demo_turns.json (greedy)
             updates the canonical artifacts logs/phase2a_baseline.*
  vectors    quick --save-vectors dump (2 turns, 16 tokens) for debugging
             logs -> logs/vec_dump.*
EOF
}

[[ $# -lt 1 ]] && { usage; exit 1; }
MODE="$1"; shift

case "$MODE" in
    repl)
        "$PY" "$PROBE" --cache-dir "$CACHE" \
            --layers 4,8,12,16 --alpha 0.85 \
            --temperature 0.6 --top-p 0.9 --max-new-tokens 64 \
            --log-prefix "$ROOT/logs/phase2a_repl" "$@"
        ;;
    baseline)
        "$PY" "$PROBE" --cache-dir "$CACHE" \
            --layers 4,8,12,16 --alpha 0.85 \
            --script "$SCRIPT" --max-new-tokens 64 \
            --log-prefix "$ROOT/logs/phase2a_baseline" "$@"
        ;;
    vectors|vec)
        "$PY" "$PROBE" --cache-dir "$CACHE" \
            --layers 4,8,12,16 \
            --script "$SCRIPT" --max-new-tokens 16 --max-turns 2 \
            --save-vectors --log-prefix "$ROOT/logs/vec_dump" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[run_probe] unknown mode '$MODE'" >&2
        usage
        exit 2
        ;;
esac