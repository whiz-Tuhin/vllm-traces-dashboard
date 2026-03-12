#!/usr/bin/env python3
"""
Preprocess vLLM JSONL traces into JSON files for the SvelteKit dashboard.
Run from the dashboard-v2/ directory:
    ../venv/bin/python preprocess.py
"""
import json
import sys
from pathlib import Path

TRACES_DIR = Path(__file__).parent.parent / "traces"
OUT_DIR = Path(__file__).parent / "static" / "data"

MODELS = {
    "Llama-2-7b": "meta-llama_Llama-2-7b-chat-hf",
    "Qwen2.5-7B": "Qwen_Qwen2.5-7B-Instruct",
}

MODES = {
    "non-streaming": {"base": TRACES_DIR, "suffix": ""},
    "streaming": {"base": TRACES_DIR / "per-token", "suffix": "-streaming"},
    "multiturn": {"base": TRACES_DIR / "multiturn", "suffix": ""},
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def process_mode(mode_key: str) -> list[dict]:
    """Returns the kv_cache rows collected (empty list for non-multiturn)."""
    cfg = MODES[mode_key]
    base = cfg["base"]
    suffix = cfg["suffix"]

    all_reqs = []
    all_fwd = []
    all_join = []
    all_kv = []

    for label, prefix in MODELS.items():
        req_path = base / f"{prefix}{suffix}.jsonl"
        fwd_path = base / f"{prefix}{suffix}_fwd.jsonl"

        if not req_path.exists():
            print(f"  SKIP {req_path} (not found)")
            continue

        req_rows = load_jsonl(req_path)
        for r in req_rows:
            r["model_label"] = label

        t0 = min(r["api_receive_ts"] for r in req_rows)
        for r in req_rows:
            r["rel_api_receive_ts"] = r["api_receive_ts"] - t0
            r["rel_engine_add_request_ts"] = r["engine_add_request_ts"] - t0
            r["rel_first_token_ts"] = r["first_token_ts"] - t0
            r["rel_completion_ts"] = r["completion_ts"] - t0
            out = max(r["output_tokens"] - 1, 0)
            r["decode_ms"] = r["tpot_ms"] * out
        all_reqs.extend(req_rows)

        if not fwd_path.exists():
            print(f"  SKIP {fwd_path} (not found)")
            continue

        fwd_rows = load_jsonl(fwd_path)
        for f in fwd_rows:
            f["batch_size"] = len(f["req_ids"])
            f["model_label"] = label
            f["rel_start_s"] = f["start_ts"] - t0
            f["rel_end_s"] = f["end_ts"] - t0

        req_lookup = {r["request_id"]: r for r in req_rows}
        for fw in fwd_rows:
            for rid in fw["req_ids"]:
                info = req_lookup.get(rid, {})
                all_join.append({
                    "fwd_id": fw["fwd_id"],
                    "request_id": rid,
                    "prompt_id": info.get("prompt_id", ""),
                    "prompt_tokens": info.get("prompt_tokens", 0),
                    "output_tokens": info.get("output_tokens", 0),
                    "tokens_in_pass": fw.get("num_tokens", {}).get(rid, 0),
                    "rel_start_s": fw["rel_start_s"],
                    "rel_end_s": fw["rel_end_s"],
                    "duration_ms": fw["duration_ms"],
                    "batch_size": fw["batch_size"],
                    "total_tokens": fw["total_tokens"],
                    "model_label": label,
                })

            if mode_key == "multiturn" and "requests" in fw:
                for rid, rinfo in fw["requests"].items():
                    all_kv.append({
                        "fwd_id": fw["fwd_id"],
                        "request_id": rid,
                        "rel_start_s": fw["rel_start_s"],
                        "duration_ms": fw["duration_ms"],
                        "is_prefill": rinfo.get("is_prefill", False),
                        "past_kv_cache_size": rinfo.get("past_kv_cache_size", 0),
                        "prefix_tokens": rinfo.get("prefix_tokens", 0),
                        "decode_tokens": rinfo.get("decode_tokens", 0),
                        "tokens_generated_so_far": rinfo.get("tokens_generated_so_far", 0),
                        "num_prompt_tokens": rinfo.get("num_prompt_tokens", 0),
                        "num_scheduled_tokens": rinfo.get("num_scheduled_tokens", 0),
                        "model_label": label,
                    })

        fwd_clean = []
        for f in fwd_rows:
            fwd_clean.append({
                "fwd_id": f["fwd_id"],
                "start_ts": f["start_ts"],
                "end_ts": f["end_ts"],
                "duration_ms": f["duration_ms"],
                "batch_size": f["batch_size"],
                "total_tokens": f["total_tokens"],
                "model_label": f["model_label"],
                "rel_start_s": f["rel_start_s"],
                "rel_end_s": f["rel_end_s"],
            })
        all_fwd.extend(fwd_clean)

    mode_dir = OUT_DIR / mode_key
    mode_dir.mkdir(parents=True, exist_ok=True)

    with open(mode_dir / "requests.json", "w") as f:
        json.dump(all_reqs, f, separators=(",", ":"))
    print(f"  {mode_key}/requests.json — {len(all_reqs)} rows")

    with open(mode_dir / "forward_passes.json", "w") as f:
        json.dump(all_fwd, f, separators=(",", ":"))
    print(f"  {mode_key}/forward_passes.json — {len(all_fwd)} rows")

    with open(mode_dir / "join.json", "w") as f:
        json.dump(all_join, f, separators=(",", ":"))
    print(f"  {mode_key}/join.json — {len(all_join)} rows")

    if all_kv:
        with open(mode_dir / "kv_cache.json", "w") as f:
            json.dump(all_kv, f, separators=(",", ":"))
        print(f"  {mode_key}/kv_cache.json — {len(all_kv)} rows")

    return all_kv


def process_per_token(mode_key: str, base: Path, suffix: str, kv_data: list[dict] | None = None):
    all_tok = []
    for label, prefix in MODELS.items():
        path = base / f"{prefix}{suffix}_per-token-timeline.jsonl"
        if not path.exists():
            print(f"  SKIP {path} (not found)")
            continue
        rows = load_jsonl(path)
        for r in rows:
            r["model_label"] = label
        all_tok.extend(rows)

    if all_tok and kv_data:
        kv_by_req: dict[str, list[dict]] = {}
        for kv in kv_data:
            kv_by_req.setdefault(kv["request_id"], []).append(kv)
        for rid in kv_by_req:
            kv_by_req[rid].sort(key=lambda k: k["fwd_id"])

        for tok in all_tok:
            rid = tok["request_id"]
            kv_rows = kv_by_req.get(rid, [])
            if not kv_rows:
                continue
            tidx = tok["token_idx"]
            if tidx < len(kv_rows):
                tok["kv_cache_size"] = kv_rows[tidx]["past_kv_cache_size"] + kv_rows[tidx]["num_scheduled_tokens"]
            elif kv_rows:
                last = kv_rows[-1]
                tok["kv_cache_size"] = last["past_kv_cache_size"] + last["num_scheduled_tokens"]

    if all_tok:
        mode_dir = OUT_DIR / mode_key
        mode_dir.mkdir(parents=True, exist_ok=True)
        with open(mode_dir / "per_token.json", "w") as f:
            json.dump(all_tok, f, separators=(",", ":"))
        print(f"  {mode_key}/per_token.json — {len(all_tok)} rows")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Preprocessing vLLM traces for SvelteKit dashboard...\n")

    kv_by_mode: dict[str, list[dict]] = {}
    for mode_key in MODES:
        print(f"[{mode_key}]")
        kv_by_mode[mode_key] = process_mode(mode_key)

    print("\n[per-token timelines]")
    process_per_token("streaming", TRACES_DIR / "per-token", "-streaming")
    process_per_token("multiturn", TRACES_DIR / "multiturn", "", kv_data=kv_by_mode.get("multiturn"))

    meta = {
        "models": list(MODELS.keys()),
        "modes": list(MODES.keys()),
        "colors": {"Llama-2-7b": "#8FB8FF", "Qwen2.5-7B": "#FF9B7A"},
        "phase_colors": {
            "Scheduling": "#F4CD78",
            "Queue/Other": "#7CD8F0",
            "Prefill": "#7ED7AB",
            "Decode": "#B495F5",
        },
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nmeta.json written")
    print("Done!")


if __name__ == "__main__":
    main()
