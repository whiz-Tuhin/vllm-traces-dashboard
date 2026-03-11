"""
vLLM Trace Comparison Dashboard
Compares Llama-2-7b-chat-hf vs Qwen2.5-7B-Instruct profiling traces.
Supports both non-streaming and streaming trace modes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

TRACES_DIR = Path(__file__).parent / "traces"

MODELS = {
    "Llama-2-7b": "meta-llama_Llama-2-7b-chat-hf",
    "Qwen2.5-7B": "Qwen_Qwen2.5-7B-Instruct",
}
COLORS   = {"Llama-2-7b": "#636EFA", "Qwen2.5-7B": "#EF553B"}
PHASE_CL = {
    "Scheduling":  "#FECB52",
    "Queue/Other": "#19D3F3",
    "Prefill":     "#00CC96",
    "Decode":      "#AB63FA",
}

st.set_page_config(page_title="vLLM Trace Comparison", layout="wide")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _note(text: str):
    with st.expander("How to read this chart", expanded=False):
        st.markdown(text)


def _ols(x: pd.Series, y: pd.Series, n=200):
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    if len(xv) < 2:
        return [], []
    c = np.polyfit(xv, yv, 1)
    xf = np.linspace(xv.min(), xv.max(), n)
    return xf, np.polyval(c, xf)


def _build_perfetto_json(label, rdf, jdf, tdf):
    """Convert traces for one model into Chrome Trace Event Format (list of dicts).

    Perfetto layout:
      - Each request is a separate "process" (pid).
      - tid 1 = Phases (Scheduling / Prefill / Decode)
      - tid 2 = Forward Passes
      - tid 3 = Token arrivals (instant events)
    """
    model_reqs = rdf[rdf["model_label"] == label].sort_values("api_receive_ts")
    if model_reqs.empty:
        return []

    global_t0 = model_reqs["api_receive_ts"].min()
    events = []

    def _us(ts):
        return (ts - global_t0) * 1_000_000

    for pid_idx, (_, req) in enumerate(model_reqs.iterrows()):
        rid = req["request_id"]
        pid = pid_idx + 1

        events.append({"ph": "M", "pid": pid, "tid": 0,
                        "name": "process_name",
                        "args": {"name": f"{req['prompt_id']} ({rid})"}})
        events.append({"ph": "M", "pid": pid, "tid": 1,
                        "name": "thread_name", "args": {"name": "Phases"}})
        events.append({"ph": "M", "pid": pid, "tid": 2,
                        "name": "thread_name", "args": {"name": "Fwd Passes"}})
        events.append({"ph": "M", "pid": pid, "tid": 3,
                        "name": "thread_name", "args": {"name": "Tokens"}})

        sched_ts  = _us(req["api_receive_ts"])
        engine_ts = _us(req["engine_add_request_ts"])
        first_ts  = _us(req["first_token_ts"])
        comp_ts   = _us(req["completion_ts"])

        events.append({"ph": "X", "pid": pid, "tid": 1, "name": "Scheduling",
                        "ts": sched_ts,  "dur": engine_ts - sched_ts,  "cat": "phase"})
        events.append({"ph": "X", "pid": pid, "tid": 1, "name": "Prefill",
                        "ts": engine_ts, "dur": first_ts - engine_ts,  "cat": "phase"})
        events.append({"ph": "X", "pid": pid, "tid": 1, "name": "Decode",
                        "ts": first_ts,  "dur": comp_ts - first_ts,   "cat": "phase"})

        # Forward passes for this request
        if jdf is not None:
            req_fwd = jdf[(jdf["model_label"] == label) &
                          (jdf["request_id"] == rid)].sort_values("rel_start_s")
            for i, (_, fw) in enumerate(req_fwd.iterrows()):
                fwd_ts  = (fw["rel_start_s"]) * 1_000_000
                fwd_dur = fw["duration_ms"] * 1_000
                events.append({
                    "ph": "X", "pid": pid, "tid": 2,
                    "name": "Prefill fwd" if i == 0 else f"Decode fwd {i}",
                    "ts": fwd_ts, "dur": fwd_dur, "cat": "fwd",
                    "args": {"fwd_id": int(fw["fwd_id"]),
                             "batch_size": int(fw["batch_size"]),
                             "tokens_in_pass": int(fw["tokens_in_pass"])}
                })

        # Per-token instants
        if tdf is not None:
            req_tok = tdf[(tdf["model_label"] == label) &
                          (tdf["request_id"] == rid)].sort_values("token_idx")
            for _, tk in req_tok.iterrows():
                tok_ts = _us(tk["timestamp"])
                events.append({
                    "ph": "i", "pid": pid, "tid": 3, "s": "t",
                    "name": f"tok_{int(tk['token_idx'])}",
                    "ts": tok_ts, "cat": "token",
                    "args": {"token_idx": int(tk["token_idx"]),
                             "itl_ms": round(tk["itl_ms"], 2)}
                })

    return events


def _parse_uploaded_perfetto(data):
    """Parse an uploaded Perfetto JSON and extract request-level metrics.

    Returns a list of dicts, one per request/process found in the trace:
      {prompt_id, scheduling_us, prefill_us, decode_us, e2e_us, tokens, itl_list}
    """
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    else:
        events = data

    procs = {}
    for ev in events:
        pid = ev.get("pid")
        if pid is None:
            continue
        if pid not in procs:
            procs[pid] = {"phases": [], "tokens": [], "fwd": [], "name": ""}
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            procs[pid]["name"] = ev.get("args", {}).get("name", "")
        elif ev.get("ph") == "X" and ev.get("cat") == "phase":
            procs[pid]["phases"].append(ev)
        elif ev.get("ph") == "i" and ev.get("cat") == "token":
            procs[pid]["tokens"].append(ev)
        elif ev.get("ph") == "X" and ev.get("cat") == "fwd":
            procs[pid]["fwd"].append(ev)

    results = []
    for pid, info in sorted(procs.items()):
        if not info["phases"]:
            continue

        phases_by_name = {p["name"]: p for p in info["phases"]}
        sched  = phases_by_name.get("Scheduling", {})
        prefill = phases_by_name.get("Prefill", {})
        decode  = phases_by_name.get("Decode", {})

        sched_dur   = sched.get("dur", 0)
        prefill_dur = prefill.get("dur", 0)
        decode_dur  = decode.get("dur", 0)
        e2e_us      = sched_dur + prefill_dur + decode_dur

        itl_list = sorted(
            [(t.get("args", {}).get("token_idx", 0), t.get("args", {}).get("itl_ms", 0))
             for t in info["tokens"]],
            key=lambda x: x[0]
        )
        decode_itls = [itl for idx, itl in itl_list if idx > 0]

        results.append({
            "prompt_id":     info["name"],
            "scheduling_ms": sched_dur / 1000,
            "prefill_ms":    prefill_dur / 1000,
            "ttft_ms":       (sched_dur + prefill_dur) / 1000,
            "decode_ms":     decode_dur / 1000,
            "e2e_ms":        e2e_us / 1000,
            "tokens":        len(info["tokens"]),
            "median_itl_ms": float(np.median(decode_itls)) if decode_itls else 0,
            "p95_itl_ms":    float(np.percentile(decode_itls, 95)) if decode_itls else 0,
            "fwd_passes":    len(info["fwd"]),
            "itl_list":      decode_itls,
        })
    return results


# ── Data loading ─────────────────────────────────────────────────────────────
MODES = {
    "Streaming":     {"base": TRACES_DIR / "per-token", "suffix": "-streaming"},
    "Non-Streaming": {"base": TRACES_DIR,               "suffix": ""},
}

@st.cache_data
def load_data(mode_key: str):
    cfg    = MODES[mode_key]
    base   = cfg["base"]
    suffix = cfg["suffix"]

    req_frames, fwd_frames, join_frames = [], [], []
    for label, prefix in MODELS.items():
        req_rows = []
        with open(base / f"{prefix}{suffix}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["model_label"] = label
                req_rows.append(r)
        req_df = pd.DataFrame(req_rows)
        t0 = req_df["api_receive_ts"].min()
        for col in ("api_receive_ts", "engine_add_request_ts", "first_token_ts", "completion_ts"):
            req_df[f"rel_{col}"] = req_df[col] - t0
        req_df["decode_ms"] = req_df["tpot_ms"] * (req_df["output_tokens"] - 1).clip(lower=0)
        req_frames.append(req_df)

        fwd_rows = []
        with open(base / f"{prefix}{suffix}_fwd.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["batch_size"]   = len(r["req_ids"])
                r["model_label"]  = label
                r["rel_start_s"]  = r["start_ts"] - t0
                r["rel_end_s"]    = r["end_ts"] - t0
                fwd_rows.append(r)
        fwd_df = pd.DataFrame(fwd_rows)
        fwd_frames.append(fwd_df)

        req_lookup = req_df.set_index("request_id")[
            ["prompt_id", "prompt_tokens", "output_tokens"]
        ].to_dict("index")
        join_rows = []
        for _, row in fwd_df.iterrows():
            for rid in row["req_ids"]:
                info = req_lookup.get(rid, {})
                join_rows.append({
                    "fwd_id":         row["fwd_id"],
                    "request_id":     rid,
                    "prompt_id":      info.get("prompt_id", ""),
                    "prompt_tokens":  info.get("prompt_tokens", 0),
                    "output_tokens":  info.get("output_tokens", 0),
                    "tokens_in_pass": row["num_tokens"].get(rid, 0),
                    "rel_start_s":    row["rel_start_s"],
                    "rel_end_s":      row["rel_end_s"],
                    "duration_ms":    row["duration_ms"],
                    "batch_size":     row["batch_size"],
                    "total_tokens":   row["total_tokens"],
                    "model_label":    label,
                })
        join_frames.append(pd.DataFrame(join_rows))

    return (pd.concat(req_frames, ignore_index=True),
            pd.concat(fwd_frames, ignore_index=True),
            pd.concat(join_frames, ignore_index=True))


@st.cache_data
def load_per_token():
    """Load per-token timeline traces (only available in per-token mode)."""
    base = TRACES_DIR / "per-token"
    frames = []
    for label, prefix in MODELS.items():
        path = base / f"{prefix}-streaming_per-token-timeline.jsonl"
        rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["model_label"] = label
                rows.append(r)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


# ── Mode selector ────────────────────────────────────────────────────────────
st.title("vLLM Trace Comparison: Llama-2-7b vs Qwen2.5-7B")

mode_col, info_col = st.columns([1, 3])
with mode_col:
    mode = st.radio(
        "Trace mode",
        list(MODES.keys()),
        index=0,
        horizontal=True,
    )
with info_col:
    if mode == "Streaming":
        st.info("**Streaming** — real per-token ITL, TTFT, and TPOT from stream events.", icon="📡")
    else:
        st.warning("**Non-Streaming** — `first_token_ts ≈ completion_ts`; decode from fwd-pass traces.", icon="📦")

is_streaming  = mode == "Streaming"
has_per_token = is_streaming
req_df, fwd_df, join_df = load_data(mode)
tok_df = load_per_token() if has_per_token else None

tab_names = [
    "Overview",
    "Forward Pass Timeline",
    "Per-Request Deep Dive",
    "Token Timeline",
    "Batching & Throughput",
    "Sanity Checks",
    "Export / Compare",
    "Fwd Pass Comparison",
    "Raw Data",
]
tab1, tab2, tab3, tab_tl, tab4, tab5, tab_ec, tab_fwd_cmp, tab6 = st.tabs(tab_names)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Summary Statistics")
    stats = []
    for label in MODELS:
        r = req_df[req_df["model_label"] == label]
        f = fwd_df[fwd_df["model_label"] == label]
        lats  = r["total_latency_ms"].dropna()
        ttfts = r["ttft_ms"].dropna()
        tpots = r["tpot_ms"].dropna()
        wall  = r["completion_ts"].max() - r["api_receive_ts"].min()
        stats.append({
            "Model":                    label,
            "Requests":                 len(r),
            "Fwd Passes":               len(f),
            "Median E2E (ms)":          f"{lats.median():.0f}",
            "p95 E2E (ms)":             f"{lats.quantile(.95):.0f}",
            "Median TTFT (ms)":         f"{ttfts.median():.1f}",
            "p95 TTFT (ms)":            f"{ttfts.quantile(.95):.1f}",
            "Median TPOT (ms)":         f"{tpots.median():.2f}",
            "p95 TPOT (ms)":            f"{tpots.quantile(.95):.2f}",
            "Wall Time (s)":            f"{wall:.1f}",
        })
    st.dataframe(pd.DataFrame(stats).set_index("Model"), use_container_width=True)
    _note("""
**E2E** = end-to-end latency (API receive → last token).  
**TTFT** = time-to-first-token (API receive → first generated token). In streaming mode this reflects
the real user-perceived delay before text starts appearing.  
**TPOT** = time-per-output-token = `(completion − first_token) / (output_tokens − 1)`.  
**Wall Time** = total real-world seconds for the whole workload.
""")

    # ── Latency CDF (E2E + TTFT side-by-side) ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("End-to-End Latency CDF")
        fig = go.Figure()
        for label in MODELS:
            v = req_df[req_df["model_label"] == label]["total_latency_ms"].sort_values().reset_index(drop=True)
            pct = [i / len(v) * 100 for i in range(len(v))]
            fig.add_trace(go.Scatter(x=v, y=pct, mode="lines", name=label,
                                     line=dict(color=COLORS[label], width=2.5),
                                     hovertemplate="%{x:.0f} ms → p%{y:.1f}<extra>" + label + "</extra>"))
            for p, tag in [(50, "p50"), (95, "p95")]:
                idx = int(p / 100 * (len(v) - 1))
                fig.add_trace(go.Scatter(
                    x=[v.iloc[idx]], y=[p], mode="markers+text",
                    marker=dict(color=COLORS[label], size=9, symbol="circle-open", line=dict(width=2)),
                    text=[f"{tag}: {v.iloc[idx]:.0f} ms"], textposition="top right",
                    textfont=dict(size=10, color=COLORS[label]), showlegend=False,
                    hovertemplate=f"{label} {tag}: {v.iloc[idx]:.0f} ms<extra></extra>"))
        for p in (50, 95):
            fig.add_hline(y=p, line_dash="dot", line_color="grey", line_width=1,
                          annotation_text=f"p{p}", annotation_position="left")
        fig.update_layout(xaxis_title="Total Latency (ms)", yaxis_title="Percentile (%)",
                          height=400, title="CDF of End-to-End Latency",
                          legend=dict(x=.65, y=.15))
        st.plotly_chart(fig, use_container_width=True)
        _note("""
The **CDF** shows what fraction of requests finish within a given latency.
- **p50 circles** = median; **p95 circles** = tail.
- A curve shifted **left** = faster model. A **steeper** curve = less variance.
""")

    with col2:
        st.subheader("Time-to-First-Token CDF")
        fig = go.Figure()
        for label in MODELS:
            v = req_df[req_df["model_label"] == label]["ttft_ms"].sort_values().reset_index(drop=True)
            pct = [i / len(v) * 100 for i in range(len(v))]
            fig.add_trace(go.Scatter(x=v, y=pct, mode="lines", name=label,
                                     line=dict(color=COLORS[label], width=2.5),
                                     hovertemplate="%{x:.1f} ms → p%{y:.1f}<extra>" + label + "</extra>"))
            for p, tag in [(50, "p50"), (95, "p95")]:
                idx = int(p / 100 * (len(v) - 1))
                fig.add_trace(go.Scatter(
                    x=[v.iloc[idx]], y=[p], mode="markers+text",
                    marker=dict(color=COLORS[label], size=9, symbol="circle-open", line=dict(width=2)),
                    text=[f"{tag}: {v.iloc[idx]:.1f} ms"], textposition="top right",
                    textfont=dict(size=10, color=COLORS[label]), showlegend=False,
                    hovertemplate=f"{label} {tag}: {v.iloc[idx]:.1f} ms<extra></extra>"))
        for p in (50, 95):
            fig.add_hline(y=p, line_dash="dot", line_color="grey", line_width=1,
                          annotation_text=f"p{p}", annotation_position="left")
        fig.update_layout(xaxis_title="TTFT (ms)", yaxis_title="Percentile (%)",
                          height=400, title="CDF of Time-to-First-Token",
                          legend=dict(x=.65, y=.15))
        st.plotly_chart(fig, use_container_width=True)
        _note("""
**TTFT** is the user-perceived delay before any text appears.  
It includes scheduling overhead + queue wait + the prefill forward pass.
Lower and steeper = better interactive responsiveness.
""")

    # ── TPOT distribution ─────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("TPOT Distribution (Violin)")
        fig = go.Figure()
        for label in MODELS:
            v = req_df[req_df["model_label"] == label]["tpot_ms"]
            fig.add_trace(go.Violin(
                y=v, name=label, box_visible=True, meanline_visible=True,
                marker_color=COLORS[label], line_color=COLORS[label],
                hovertemplate="%{y:.2f} ms<extra>" + label + "</extra>"))
        fig.update_layout(yaxis_title="TPOT (ms)", height=380,
                          title="Per-Token Decode Latency Distribution", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        _note("""
**TPOT** (Time Per Output Token) = average decode-phase time per generated token.
- The violin width shows density; the inner box is IQR; the line is mean.
- Lower TPOT = faster token generation = better streaming UX.
- Spread comes from varying batch sizes and KV-cache pressure across requests.
""")

    with col4:
        st.subheader("Forward Pass Duration (Box)")
        fig = go.Figure()
        for label in MODELS:
            v = fwd_df[fwd_df["model_label"] == label]["duration_ms"]
            fig.add_trace(go.Box(y=v, name=label, marker_color=COLORS[label],
                                 boxmean="sd", boxpoints="outliers", jitter=.3, pointpos=-1.8,
                                 hovertemplate="%{y:.1f} ms<extra>" + label + "</extra>"))
        fig.update_layout(yaxis_title="Duration (ms)", yaxis_type="log", height=380,
                          title="GPU Forward-Pass Duration", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        _note("""
Each box summarises **all** forward passes (prefill + decode).
- Log y-axis: prefill passes (top) are 10–100× longer than decode passes (bottom cluster).
- **Box** = IQR (25th–75th percentile); diamond = mean ± σ; dots = outliers.
""")

    # ── Phase breakdown stacked bar ───────────────────────────────────────────
    st.subheader("Median Latency Phase Breakdown")
    phase_rows = []
    for label in MODELS:
        r   = req_df[req_df["model_label"] == label]
        e2e = r["total_latency_ms"].median()

        if is_streaming:
            sched   = r["scheduling_overhead_ms"].median()
            ttft    = r["ttft_ms"].median()
            prefill = ttft - sched
            decode  = r["decode_ms"].median()
            queue   = max(e2e - sched - prefill - decode, 0)
        else:
            jdf = join_df[join_df["model_label"] == label].sort_values("rel_start_s")
            grp = jdf.groupby("request_id", group_keys=False)
            sched   = r["scheduling_overhead_ms"].median()
            prefill = grp["duration_ms"].first().median()
            decode  = grp["duration_ms"].apply(lambda s: s.iloc[1:].sum()).median()
            queue   = max(e2e - sched - prefill - decode, 0)

        phase_rows += [
            {"Model": label, "Phase": "Scheduling",  "ms": sched},
            {"Model": label, "Phase": "Queue/Other", "ms": queue},
            {"Model": label, "Phase": "Prefill",     "ms": prefill},
            {"Model": label, "Phase": "Decode",      "ms": decode},
        ]

    pdf = pd.DataFrame(phase_rows)
    fig = px.bar(pdf, x="Model", y="ms", color="Phase", barmode="stack",
                 color_discrete_map=PHASE_CL, text_auto=".0f", height=420,
                 title="Where Time Goes (Median per Request)",
                 labels={"ms": "Time (ms)"},
                 category_orders={"Phase": list(PHASE_CL)})
    fig.update_traces(textposition="inside", textfont_size=13)
    fig.update_layout(legend=dict(orientation="h", y=1.07))
    st.plotly_chart(fig, use_container_width=True)
    _note(f"""
| Phase | Colour | Meaning |
|---|---|---|
| **Scheduling** | Yellow | HTTP/Python overhead before the engine enqueues the request. |
| **Queue/Other** | Cyan | Time waiting in the scheduler queue + inter-pass gaps. |
| **Prefill** | Green | {'TTFT minus scheduling overhead (request-level).' if is_streaming else 'First forward pass GPU time (from fwd traces).'} |
| **Decode** | Purple | {'`tpot × (output_tokens − 1)` (request-level).' if is_streaming else 'Sum of all subsequent fwd-pass GPU times.'} |

The bars sum to ≈ median E2E latency.  
**Tall Decode** → output-bound; **Tall Prefill** → prompt-bound; **Tall Queue** → high concurrency / KV-cache pressure.
""")

    # ── Histogram ─────────────────────────────────────────────────────────────
    st.subheader("E2E Latency Histogram")
    fig = px.histogram(req_df, x="total_latency_ms", color="model_label",
                       barmode="overlay", nbins=50, opacity=.6,
                       color_discrete_map=COLORS,
                       labels={"total_latency_ms": "E2E Latency (ms)", "model_label": "Model"},
                       height=320, title="Request Count per Latency Bucket")
    for label in MODELS:
        med = req_df[req_df["model_label"] == label]["total_latency_ms"].median()
        fig.add_vline(x=med, line_dash="dash", line_color=COLORS[label],
                      annotation_text=f"{label} med", annotation_position="top right",
                      annotation_font_color=COLORS[label])
    fig.update_layout(legend=dict(x=.75, y=.9))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
Bar height = number of requests in that latency bucket.
- **Narrow peak** = consistent latency. **Long right tail** = slow outliers.
- Dashed line = median per model.
""")

    # ── Per-token ITL overview (only in per-token mode) ───────────────────────
    if has_per_token and tok_df is not None:
        st.divider()
        st.subheader("Inter-Token Latency (ITL) — All Requests")
        st.markdown(
            "Measured from real per-token stream timestamps. "
            "`token_idx=0` is the first generated token (after prefill); "
            "ITL for subsequent tokens is the wall-clock gap to the previous token."
        )

        decode_tok = tok_df[tok_df["token_idx"] > 0].copy()

        col_itl1, col_itl2 = st.columns(2)

        with col_itl1:
            fig = go.Figure()
            for label in MODELS:
                v = decode_tok[decode_tok["model_label"] == label]["itl_ms"].sort_values().reset_index(drop=True)
                pct = [i / len(v) * 100 for i in range(len(v))]
                fig.add_trace(go.Scatter(
                    x=v, y=pct, mode="lines", name=label,
                    line=dict(color=COLORS[label], width=2.5),
                    hovertemplate="%{x:.1f} ms → p%{y:.1f}<extra>" + label + "</extra>"))
                for p, tag in [(50, "p50"), (95, "p95"), (99, "p99")]:
                    idx = int(p / 100 * (len(v) - 1))
                    fig.add_trace(go.Scatter(
                        x=[v.iloc[idx]], y=[p], mode="markers+text",
                        marker=dict(color=COLORS[label], size=8, symbol="circle-open", line=dict(width=2)),
                        text=[f"{tag}: {v.iloc[idx]:.1f} ms"], textposition="top right",
                        textfont=dict(size=9, color=COLORS[label]), showlegend=False,
                        hovertemplate=f"{label} {tag}: {v.iloc[idx]:.1f} ms<extra></extra>"))
            for p in (50, 95, 99):
                fig.add_hline(y=p, line_dash="dot", line_color="grey", line_width=1,
                              annotation_text=f"p{p}", annotation_position="left")
            fig.update_layout(xaxis_title="ITL (ms)", yaxis_title="Percentile (%)",
                              height=400, title="CDF of Inter-Token Latency (all decode tokens)",
                              legend=dict(x=.65, y=.15))
            st.plotly_chart(fig, use_container_width=True)
            _note("""
**ITL CDF** — each data point is one *individual* token's inter-token latency.
- p50 = the median streaming "stutter" between tokens.
- p95/p99 = the tail — tokens that took much longer (often due to a new prefill being scheduled mid-decode, or a batch-size change).
- A tight curve (steep) means consistent token delivery; a long right tail means occasional jitter.
""")

        with col_itl2:
            fig = px.histogram(
                decode_tok, x="itl_ms", color="model_label",
                barmode="overlay", nbins=80, opacity=.6,
                color_discrete_map=COLORS,
                labels={"itl_ms": "ITL (ms)", "model_label": "Model"},
                height=400, title="ITL Distribution (all decode tokens)")
            for label in MODELS:
                med = decode_tok[decode_tok["model_label"] == label]["itl_ms"].median()
                fig.add_vline(x=med, line_dash="dash", line_color=COLORS[label],
                              annotation_text=f"p50={med:.1f}", annotation_position="top right",
                              annotation_font_color=COLORS[label])
            fig.update_layout(legend=dict(x=.75, y=.9))
            st.plotly_chart(fig, use_container_width=True)
            _note("""
Histogram of every individual token's inter-token latency.
The sharp peak is the steady-state decode cadence; the right tail captures jitter from prefill interruptions and batch changes.
""")

        # ── ITL heatmap: token position vs ITL ────────────────────────────────
        st.subheader("ITL vs Token Position (Heatmap)")
        st.markdown("Does ITL change as generation progresses? Each row is a token position, binned by ITL value.")
        for label in MODELS:
            dt = decode_tok[decode_tok["model_label"] == label]
            max_idx = int(dt["token_idx"].quantile(.95))
            dt_clip = dt[dt["token_idx"] <= max_idx]
            fig = go.Figure(go.Histogram2d(
                x=dt_clip["token_idx"], y=dt_clip["itl_ms"],
                colorscale="Blues" if label == "Llama-2-7b" else "Reds",
                nbinsx=min(max_idx, 64), nbinsy=50,
                hovertemplate="Token #%{x}<br>ITL: %{y:.1f} ms<br>Count: %{z}<extra>" + label + "</extra>"))
            fig.update_layout(
                xaxis_title="Output Token Index",
                yaxis_title="ITL (ms)",
                height=350,
                title=f"{label} — ITL Heatmap by Token Position")
            st.plotly_chart(fig, use_container_width=True)
        _note("""
Each cell's colour intensity = number of tokens at that (token_index, ITL) combination across all requests.
- A **horizontal bright band** means ITL is stable regardless of position → memory-bound decode (consistent).
- **Brightening at early positions** means the first few tokens after prefill are slower (batch ramp-up).
- **Vertical streaks at specific positions** may indicate periodic KV-cache eviction or scheduler pauses.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Forward Pass Timeline
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Forward Pass Duration Over Time")
    st.caption("Each marker = one GPU forward pass.  Size ∝ batch size.")
    fig = go.Figure()
    for label in MODELS:
        f = fwd_df[fwd_df["model_label"] == label]
        fig.add_trace(go.Scatter(
            x=f["rel_start_s"], y=f["duration_ms"], mode="markers", name=label,
            marker=dict(color=COLORS[label], size=f["batch_size"] * 1.5 + 3, opacity=.55, line=dict(width=0)),
            customdata=list(zip(f["fwd_id"], f["batch_size"], f["total_tokens"], f["duration_ms"].round(1))),
            hovertemplate="<b>" + label + "</b><br>Time: %{x:.2f}s<br>Duration: %{customdata[3]:.1f} ms<br>"
                          "Batch: %{customdata[1]}<br>Tokens: %{customdata[2]}<br>fwd_id: %{customdata[0]}<extra></extra>"))
    fig.update_layout(xaxis_title="Relative Time (s)", yaxis_title="Duration (ms)",
                      yaxis_type="log", height=440,
                      title="GPU Time per Forward Pass (log scale)", legend=dict(x=.01, y=.99))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
**Two horizontal bands** are visible:
- **Upper band (10–1000 ms)** = prefill passes (processing the full prompt).
- **Lower band (1–10 ms)** = decode passes (generating one token per request in the batch).

Bigger markers = more requests batched together.
""")

    st.subheader("Batch Size Over Time")
    roll_w = st.slider("Rolling-average window (passes)", 1, 50, 20, key="roll2")
    fig = go.Figure()
    for label in MODELS:
        f = fwd_df[fwd_df["model_label"] == label].sort_values("rel_start_s")
        fig.add_trace(go.Scatter(x=f["rel_start_s"], y=f["batch_size"],
                                 mode="lines", name=label + " raw",
                                 line=dict(color=COLORS[label], width=1, dash="dot"), opacity=.4))
        fig.add_trace(go.Scatter(x=f["rel_start_s"],
                                 y=f["batch_size"].rolling(roll_w, min_periods=1).mean(),
                                 mode="lines", name=label + " avg",
                                 line=dict(color=COLORS[label], width=2.5)))
    fig.update_layout(xaxis_title="Relative Time (s)", yaxis_title="Requests in Batch",
                      height=360, title="Continuous Batching Occupancy",
                      legend=dict(x=.01, y=.99))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
Higher batch = better GPU utilisation. The rolling average smooths noise.
Batches ramp up as requests arrive and taper off as the workload completes.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Per-Request Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("Select a **prompt ID** to inspect how both models processed the same input.")
    prompt_ids = sorted(req_df["prompt_id"].dropna().unique())
    sel_idx = min(10, len(prompt_ids) - 1)
    selected = st.selectbox("Prompt ID", prompt_ids, index=sel_idx)

    # ── Lifecycle Gantt with forward-pass tick marks ──────────────────────────
    st.subheader(f"Request Lifecycle — {selected}")

    phases = [
        ("Scheduling", "rel_api_receive_ts",       "rel_engine_add_request_ts", PHASE_CL["Scheduling"]),
        ("Prefill",    "rel_engine_add_request_ts", "rel_first_token_ts",        PHASE_CL["Prefill"]),
        ("Decode",     "rel_first_token_ts",        "rel_completion_ts",          PHASE_CL["Decode"]),
    ]

    fig = go.Figure()
    model_labels = list(MODELS.keys())
    for i, label in enumerate(model_labels):
        r = req_df[(req_df["model_label"] == label) & (req_df["prompt_id"] == selected)]
        if r.empty:
            continue
        r = r.iloc[0]
        for phase_name, sc, ec, color in phases:
            s, e = r.get(sc, 0), r.get(ec, 0)
            dur = (e - s) * 1000
            fig.add_trace(go.Bar(
                x=[e - s], y=[label], base=[s], orientation="h",
                name=phase_name, marker_color=color,
                showlegend=(i == 0),
                text=f"{dur:.0f} ms" if dur >= 1 else "",
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=11),
                hovertemplate=f"<b>{phase_name}</b><br>{dur:.0f} ms<extra>{label}</extra>"))

        # Forward-pass tick marks
        sub_fwd = join_df[(join_df["model_label"] == label) &
                          (join_df["prompt_id"] == selected)].sort_values("rel_start_s")
        if not sub_fwd.empty:
            fig.add_trace(go.Scatter(
                x=sub_fwd["rel_start_s"],
                y=[label] * len(sub_fwd),
                mode="markers",
                name="Fwd passes" if i == 0 else None,
                showlegend=(i == 0),
                marker=dict(symbol="line-ns", size=14, line=dict(width=2, color="black"), color="black"),
                customdata=list(zip(sub_fwd["fwd_id"], sub_fwd["duration_ms"].round(1),
                                    sub_fwd["batch_size"], sub_fwd["tokens_in_pass"])),
                hovertemplate="<b>Fwd pass</b><br>Time: %{x:.3f}s<br>Duration: %{customdata[1]} ms<br>"
                              "Batch: %{customdata[2]}<br>Tokens: %{customdata[3]}<br>"
                              "fwd_id: %{customdata[0]}<extra>" + label + "</extra>"))

    fig.update_layout(
        barmode="stack", xaxis_title="Relative Time (s)", yaxis_title="",
        height=300, legend=dict(orientation="h", y=1.20),
        title=f"Pipeline Phases for '{selected}' — black ticks = individual forward passes")
    st.plotly_chart(fig, use_container_width=True)
    _note("""
| Colour | Phase | Description |
|---|---|---|
| **Yellow** | Scheduling | HTTP/Python before the engine enqueues the request. |
| **Green** | Prefill | Queue wait + prefill forward pass (prompt processing). |
| **Purple** | Decode | Token-by-token generation until completion. |

**Black tick marks** show exactly when each forward pass was invoked for this request.
The first tick (inside or near the green bar) is the prefill pass; subsequent ticks are decode passes.
Hover over a tick to see its batch size, duration, and token count.
""")

    # ── Latency detail metrics for selected prompt ────────────────────────────
    detail_rows = []
    for label in MODELS:
        r = req_df[(req_df["model_label"] == label) & (req_df["prompt_id"] == selected)]
        if r.empty:
            continue
        r = r.iloc[0]
        detail_rows.append({
            "Model":         label,
            "Prompt Tokens":  int(r["prompt_tokens"]),
            "Output Tokens":  int(r["output_tokens"]),
            "Sched (ms)":     round(r["scheduling_overhead_ms"], 1),
            "TTFT (ms)":      round(r["ttft_ms"], 1),
            "TPOT (ms)":      round(r["tpot_ms"], 2),
            "Decode (ms)":    round(r["decode_ms"], 1),
            "E2E (ms)":       round(r["total_latency_ms"], 1),
        })
    if detail_rows:
        st.dataframe(pd.DataFrame(detail_rows).set_index("Model"), use_container_width=True)

    # ── Per-prompt forward-pass scatter ────────────────────────────────────────
    st.subheader(f"Forward Passes for '{selected}'")
    sub = join_df[join_df["prompt_id"] == selected].copy()
    if sub.empty:
        st.warning("No forward-pass data for this prompt.")
    else:
        fig = go.Figure()
        for label in MODELS:
            s = sub[sub["model_label"] == label].sort_values("rel_start_s")
            if s.empty:
                continue
            is_pf = s["fwd_id"] == s["fwd_id"].iloc[0]
            fig.add_trace(go.Scatter(
                x=s["rel_start_s"], y=s["duration_ms"], mode="markers+lines", name=label,
                line=dict(color=COLORS[label], width=1, dash="dot"),
                marker=dict(color=COLORS[label],
                            size=np.where(is_pf, 14, 7),
                            symbol=np.where(is_pf, "star", "circle"),
                            opacity=.85, line=dict(width=1, color="white")),
                customdata=list(zip(s["fwd_id"], s["batch_size"],
                                    s["tokens_in_pass"], s["total_tokens"],
                                    s["duration_ms"].round(1))),
                hovertemplate="<b>" + label + "</b><br>Time: %{x:.3f}s<br>Duration: %{customdata[4]:.1f} ms<br>"
                              "Tokens (this req): %{customdata[2]}<br>Batch: %{customdata[1]}<br>"
                              "Total batch tokens: %{customdata[3]}<br>fwd_id: %{customdata[0]}<extra></extra>"))
        fig.update_layout(xaxis_title="Relative Time (s)", yaxis_title="Fwd Pass Duration (ms)",
                          yaxis_type="log", height=420,
                          title=f"GPU Duration per Forward Pass — ★ = prefill, ● = decode",
                          legend=dict(x=.01, y=.99))
        st.plotly_chart(fig, use_container_width=True)
        _note("""
The **star marker** (★) is the prefill pass; **circles** (●) are decode passes.
A dotted line connects passes in order so you can see the decode cadence.
Log y-axis keeps both prefill (large) and decode (small) visible simultaneously.
""")

        st.dataframe(
            sub[["model_label", "fwd_id", "rel_start_s", "duration_ms",
                 "tokens_in_pass", "batch_size", "total_tokens"]]
            .rename(columns={"model_label": "Model", "fwd_id": "Fwd ID",
                             "rel_start_s": "Start (s)", "duration_ms": "Duration (ms)",
                             "tokens_in_pass": "Tokens (req)", "batch_size": "Batch",
                             "total_tokens": "Batch Tokens"})
            .sort_values(["Model", "Fwd ID"])
            .style.format({"Start (s)": "{:.3f}", "Duration (ms)": "{:.1f}"}),
            use_container_width=True, height=300)

    # ── Per-token timeline ────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Per-Token Timeline — '{selected}'")

    if has_per_token and tok_df is not None:
        # ── REAL per-token data ───────────────────────────────────────────
        st.markdown(
            "**Real per-token timestamps** recorded at the streaming client. "
            "`itl_ms` = wall-clock gap since the previous token."
        )

        # Get the per-token data for the selected prompt, plus fwd-pass data for correlation
        sel_req_ids = req_df[req_df["prompt_id"] == selected]["request_id"].values

        for label in MODELS:
            r = req_df[(req_df["model_label"] == label) & (req_df["prompt_id"] == selected)]
            if r.empty:
                continue
            r = r.iloc[0]
            rid = r["request_id"]
            pt = tok_df[(tok_df["model_label"] == label) &
                        (tok_df["request_id"] == rid)].sort_values("token_idx").copy()
            if pt.empty:
                continue

            fwd_for_req = join_df[(join_df["model_label"] == label) &
                                  (join_df["request_id"] == rid)].sort_values("rel_start_s")

            st.markdown(f"#### {label}")
            mc1, mc2, mc3, mc4 = st.columns(4)
            decode_pt = pt[pt["token_idx"] > 0]
            mc1.metric("Tokens", len(pt))
            mc2.metric("Median ITL", f"{decode_pt['itl_ms'].median():.1f} ms")
            mc3.metric("p95 ITL", f"{decode_pt['itl_ms'].quantile(.95):.1f} ms")
            mc4.metric("Max ITL", f"{decode_pt['itl_ms'].max():.1f} ms")

            # ── ITL line chart with fwd-pass overlay ──────────────────────
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # ITL line
            fig.add_trace(go.Scatter(
                x=pt["token_idx"], y=pt["itl_ms"],
                mode="lines", name="ITL (ms)",
                line=dict(color=COLORS[label], width=1.5),
                hovertemplate="Token #%{x}<br>ITL: %{y:.2f} ms<extra></extra>"),
                secondary_y=False)

            avg_itl = decode_pt["itl_ms"].mean()
            fig.add_hline(y=avg_itl, line_dash="dash", line_color=COLORS[label],
                          line_width=1, secondary_y=False,
                          annotation_text=f"avg ITL: {avg_itl:.1f} ms",
                          annotation_position="right",
                          annotation_font_color=COLORS[label])

            # Cumulative token arrival timeline
            fig.add_trace(go.Scatter(
                x=pt["token_idx"], y=pt["rel_ts_ms"],
                mode="lines", name="Cumulative time (ms)",
                line=dict(color="grey", width=1, dash="dot"),
                hovertemplate="Token #%{x}<br>Time since start: %{y:.1f} ms<extra></extra>"),
                secondary_y=True)

            # Forward-pass tick marks on the ITL chart
            if not fwd_for_req.empty:
                fwd_decode = fwd_for_req.iloc[1:]
                if not fwd_decode.empty:
                    fwd_tok_idx = list(range(1, len(fwd_decode) + 1))
                    fig.add_trace(go.Scatter(
                        x=fwd_tok_idx, y=fwd_decode["duration_ms"],
                        mode="markers", name="Fwd pass GPU (ms)",
                        marker=dict(color="black", size=4, symbol="cross", opacity=.5),
                        customdata=list(zip(fwd_decode["fwd_id"], fwd_decode["batch_size"])),
                        hovertemplate="Token #%{x}<br>GPU: %{y:.1f} ms<br>"
                                      "Batch: %{customdata[1]}<br>fwd_id: %{customdata[0]}<extra></extra>"),
                        secondary_y=False)

            fig.update_layout(
                xaxis_title="Output Token Index",
                height=400,
                title=f"{label} — ITL per Token (line) + GPU fwd duration (crosses) + Cumulative Time (dotted)",
                legend=dict(x=.01, y=.99))
            fig.update_yaxes(title_text="Latency (ms)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Time (ms)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # ── Zoomed ITL with batch size correlation ────────────────────
            if not fwd_for_req.empty and len(fwd_for_req) > 1:
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(
                    x=decode_pt["token_idx"], y=decode_pt["itl_ms"],
                    mode="lines", name="ITL (ms)",
                    line=dict(color=COLORS[label], width=1.5),
                    fill="tozeroy",
                    fillcolor=COLORS[label].replace(")", ", 0.1)").replace("rgb", "rgba")
                        if "rgb" in COLORS[label] else COLORS[label],
                    hovertemplate="Token #%{x}<br>ITL: %{y:.2f} ms<extra></extra>"),
                    secondary_y=False)

                fwd_decode = fwd_for_req.iloc[1:]
                if not fwd_decode.empty:
                    fwd_tok_idx = list(range(1, len(fwd_decode) + 1))
                    fig2.add_trace(go.Scatter(
                        x=fwd_tok_idx, y=fwd_decode["batch_size"].values,
                        mode="lines+markers", name="Batch size",
                        line=dict(color="black", width=1.5, dash="dash"),
                        marker=dict(size=3, color="black"),
                        hovertemplate="Token #%{x}<br>Batch: %{y}<extra></extra>"),
                        secondary_y=True)

                fig2.update_layout(
                    xaxis_title="Output Token Index (decode only)",
                    height=350,
                    title=f"{label} — ITL vs Batch Size (do larger batches slow down token delivery?)",
                    legend=dict(x=.01, y=.99))
                fig2.update_yaxes(title_text="ITL (ms)", secondary_y=False)
                fig2.update_yaxes(title_text="Batch Size", secondary_y=True)
                st.plotly_chart(fig2, use_container_width=True)

        _note("""
**ITL line (coloured)**: the real wall-clock inter-token latency for each generated token.
This is what the end-user experiences while watching the stream.

**Crosses (+)**: GPU forward-pass duration for the decode pass that generated this token.
The gap between the ITL line and the crosses is scheduler + Python overhead between passes.

**Dotted grey line**: cumulative time since the first token — lets you see total decode progress.

**Bottom chart — ITL vs Batch Size**: overlays batch size (dashed black) on ITL.
- When batch size rises (new requests join), ITL often increases slightly — more work per pass.
- When batch size drops (requests complete), ITL decreases.
- A strong visual correlation confirms the model is throughput-limited at that batch size.
""")

    else:
        st.info("Per-token timeline is available in **Streaming** mode. Switch using the radio button above.")


# ═══════════════════════════════════════════════════════════════════════════════
# Token Timeline Tab — Gantt-style per-token view
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tl:
    st.subheader("Token Timeline")
    st.markdown(
        "Select a prompt to see a **Gantt-style timeline** of its entire lifecycle for both models side by side. "
        "Each chart has three swim lanes: request phases, GPU forward passes, and individual token arrivals."
    )

    tl_prompt_ids = sorted(req_df["prompt_id"].dropna().unique())
    tl_sel_idx = min(10, len(tl_prompt_ids) - 1)
    tl_selected = st.selectbox("Prompt ID", tl_prompt_ids, index=tl_sel_idx, key="tl_prompt")

    if not has_per_token or tok_df is None:
        st.info("Token Timeline requires **Streaming** mode. Switch using the radio button above.")
    else:
        def _build_gantt(label, selected_prompt):
            """Build a 3-lane Gantt figure for one model + one prompt."""
            r = req_df[(req_df["model_label"] == label) & (req_df["prompt_id"] == selected_prompt)]
            if r.empty:
                return None, None
            r = r.iloc[0]
            rid = r["request_id"]

            pt = tok_df[(tok_df["model_label"] == label) &
                        (tok_df["request_id"] == rid)].sort_values("token_idx")
            fwd = join_df[(join_df["model_label"] == label) &
                          (join_df["request_id"] == rid)].sort_values("rel_start_s")

            if pt.empty:
                return None, None

            # Time anchor: api_receive_ts for this request
            t0 = r["api_receive_ts"]
            sched_start = 0
            sched_end   = (r["engine_add_request_ts"] - t0) * 1000
            prefill_end = (r["first_token_ts"] - t0) * 1000
            decode_end  = (r["completion_ts"] - t0) * 1000

            # Y-axis lanes (bottom to top)
            lanes = ["Tokens", "Fwd Passes", "Phases"]

            fig = go.Figure()

            # ── Lane 1: Phases ────────────────────────────────────────────
            phase_data = [
                ("Scheduling", sched_start, sched_end,   PHASE_CL["Scheduling"]),
                ("Prefill",    sched_end,   prefill_end, PHASE_CL["Prefill"]),
                ("Decode",     prefill_end, decode_end,  PHASE_CL["Decode"]),
            ]
            for pname, ps, pe, pcol in phase_data:
                dur = pe - ps
                fig.add_trace(go.Bar(
                    x=[dur], y=["Phases"], base=[ps], orientation="h",
                    name=pname, marker_color=pcol, width=0.6,
                    text=f"{pname} ({dur:.0f} ms)" if dur > decode_end * 0.03 else "",
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(size=10, color="white"),
                    showlegend=False,
                    hovertemplate=f"<b>{pname}</b><br>{dur:.0f} ms<extra></extra>"))

            # ── Lane 2: Forward Passes ────────────────────────────────────
            if not fwd.empty:
                for i, (_, fw) in enumerate(fwd.iterrows()):
                    fwd_start_ms = (fw["rel_start_s"]) * 1000
                    fwd_dur      = fw["duration_ms"]
                    is_prefill   = (i == 0)
                    fcol = PHASE_CL["Prefill"] if is_prefill else PHASE_CL["Decode"]
                    fig.add_trace(go.Bar(
                        x=[fwd_dur], y=["Fwd Passes"], base=[fwd_start_ms],
                        orientation="h", marker_color=fcol, width=0.6,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{'Prefill' if is_prefill else 'Decode'} fwd</b><br>"
                            f"Start: {fwd_start_ms:.1f} ms<br>"
                            f"Duration: {fwd_dur:.1f} ms<br>"
                            f"Batch size: {fw['batch_size']}<br>"
                            f"Tokens: {fw['tokens_in_pass']}<br>"
                            f"fwd_id: {fw['fwd_id']}<extra></extra>"
                        )))

            # ── Lane 3: Token markers ─────────────────────────────────────
            decode_itl = pt[pt["token_idx"] > 0]["itl_ms"]
            if not decode_itl.empty:
                med_itl = decode_itl.median()
                p95_itl = decode_itl.quantile(0.95)
            else:
                med_itl, p95_itl = 0, 0

            # Offset token rel_ts_ms to account for TTFT
            # token rel_ts_ms is relative to first_token_ts, so add prefill_end
            token_x  = pt["rel_ts_ms"].values + prefill_end
            token_itl = pt["itl_ms"].values

            colors = []
            for idx, itl_val in enumerate(token_itl):
                if idx == 0:
                    colors.append(PHASE_CL["Prefill"])
                elif itl_val <= med_itl:
                    colors.append("#2ECC71")
                elif itl_val <= p95_itl:
                    colors.append("#F39C12")
                else:
                    colors.append("#E74C3C")

            fig.add_trace(go.Scatter(
                x=token_x,
                y=["Tokens"] * len(token_x),
                mode="markers",
                marker=dict(color=colors, size=6, symbol="line-ns",
                            line=dict(width=2, color=colors)),
                showlegend=False,
                customdata=list(zip(pt["token_idx"], pt["itl_ms"].round(2), pt["rel_ts_ms"].round(1))),
                hovertemplate=(
                    "Token #%{customdata[0]}<br>"
                    "ITL: %{customdata[1]} ms<br>"
                    "Time: %{x:.1f} ms<extra></extra>"
                )))

            fig.update_layout(
                barmode="stack",
                xaxis_title="Time since request start (ms)",
                yaxis=dict(categoryorder="array", categoryarray=lanes),
                height=280,
                margin=dict(l=80, r=20, t=10, b=50),
                showlegend=False,
            )

            # Summary metrics
            metrics = {
                "Tokens": int(r["output_tokens"]),
                "TTFT": f"{r['ttft_ms']:.1f} ms",
                "Median ITL": f"{med_itl:.1f} ms",
                "p95 ITL": f"{p95_itl:.1f} ms",
                "Max ITL": f"{decode_itl.max():.1f} ms" if not decode_itl.empty else "–",
                "E2E": f"{r['total_latency_ms']:.0f} ms",
            }
            return fig, metrics

        col_l, col_div, col_r = st.columns([20, 1, 20])
        model_list = list(MODELS.keys())

        with col_div:
            st.markdown(
                "<div style='border-left:2px solid #444; height:420px; margin:auto;'></div>",
                unsafe_allow_html=True,
            )

        for col, label in zip([col_l, col_r], model_list):
            with col:
                fig, metrics = _build_gantt(label, tl_selected)
                if fig is None:
                    st.warning(f"No data for {label} / {tl_selected}")
                    continue
                color = COLORS[label]
                st.markdown(
                    f"<h3 style='margin-bottom:0; color:{color};'>{label}</h3>",
                    unsafe_allow_html=True,
                )
                metric_html = "".join(
                    f"<span style='margin-right:18px;'>"
                    f"<span style='color:#999; font-size:0.78rem;'>{k}</span><br>"
                    f"<span style='font-size:1.05rem; font-weight:600;'>{v}</span>"
                    f"</span>"
                    for k, v in metrics.items()
                )
                st.markdown(
                    f"<div style='display:flex; flex-wrap:wrap; gap:4px 0; margin:4px 0 10px;'>{metric_html}</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Colour legend
        leg1, leg2, leg3 = st.columns(3)
        leg1.markdown("**Phases:** 🟡 Scheduling  🟢 Prefill  🟣 Decode")
        leg2.markdown("**Fwd Passes:** 🟢 Prefill fwd  🟣 Decode fwd")
        leg3.markdown("**Tokens:** 🟢 Fast (≤ p50)  🟡 Normal (p50–p95)  🔴 Slow (> p95)")

        _note("""
**Three swim lanes per model:**

- **Phases** (top) — the three lifecycle stages as coloured bars: Scheduling (yellow), Prefill (green), Decode (purple).
- **Fwd Passes** (middle) — each GPU forward pass as a bar. The first (wide) bar is the prefill pass; subsequent narrow bars are decode passes. Width = GPU execution time. Hover for batch size and token count.
- **Tokens** (bottom) — each tick mark is one generated token, placed at the exact time it was emitted to the client. Colour indicates speed: green = fast (ITL ≤ median), yellow = normal (median–p95), red = slow (> p95).

**How to read it:**
- Dense, green tick marks = smooth, fast streaming.
- Gaps or red ticks = the user experienced a stutter (usually caused by a prefill for another request or a batch-size change).
- Compare left (Llama) vs right (Qwen) to see which model delivers tokens more consistently.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Batching & Throughput
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Batch Size Distribution")
        fig = go.Figure()
        for label in MODELS:
            f = fwd_df[fwd_df["model_label"] == label]
            fig.add_trace(go.Histogram(x=f["batch_size"], name=label,
                                       marker_color=COLORS[label], opacity=.72,
                                       autobinx=False, xbins=dict(start=.5, end=17.5, size=1)))
            m = f["batch_size"].mean()
            fig.add_vline(x=m, line_dash="dash", line_color=COLORS[label],
                          annotation_text=f"mean={m:.1f}", annotation_position="top right",
                          annotation_font_color=COLORS[label])
        fig.update_layout(barmode="overlay", xaxis_title="Requests per Fwd Pass",
                          yaxis_title="Count", height=380,
                          title="Batch Size Frequency", legend=dict(x=.65, y=.9))
        st.plotly_chart(fig, use_container_width=True)
        _note("Higher batch sizes = better GPU utilisation. Dashed lines = mean batch size.")

    with col2:
        st.subheader("Token Throughput Over Time")
        bsz = st.slider("Time bin (s)", .5, 5., 1., .5, key="tput_bin")
        fig = go.Figure()
        for label in MODELS:
            f = fwd_df[fwd_df["model_label"] == label].copy()
            f["tb"] = (f["rel_start_s"] // bsz) * bsz
            tp = f.groupby("tb")["total_tokens"].sum() / bsz
            fig.add_trace(go.Scatter(x=tp.index, y=tp.values, mode="lines+markers", name=label,
                                     line=dict(color=COLORS[label], width=2),
                                     marker=dict(size=4),
                                     hovertemplate="Time: %{x:.1f}s<br>Tokens/s: %{y:.0f}<extra>" + label + "</extra>"))
        fig.update_layout(xaxis_title="Relative Time (s)", yaxis_title="Tokens / s",
                          height=380, title="Token Generation Rate",
                          legend=dict(x=.65, y=.9))
        st.plotly_chart(fig, use_container_width=True)
        _note("Tokens/s includes both prefill and decode tokens processed in each time window.")

    # ── Prompt tokens vs E2E ──────────────────────────────────────────────────
    st.subheader("Prompt Length vs E2E Latency")
    fig = go.Figure()
    for label in MODELS:
        s = req_df[req_df["model_label"] == label]
        fig.add_trace(go.Scatter(
            x=s["prompt_tokens"], y=s["total_latency_ms"], mode="markers", name=label,
            marker=dict(color=COLORS[label], opacity=.5, size=7),
            customdata=list(zip(s["prompt_id"], s["output_tokens"])),
            hovertemplate="<b>" + label + "</b><br>Prompt: %{x} tok<br>E2E: %{y:.0f} ms<br>"
                          "Output: %{customdata[1]} tok<br>ID: %{customdata[0]}<extra></extra>"))
        xf, yf = _ols(s["prompt_tokens"], s["total_latency_ms"])
        if len(xf):
            fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines", name=label + " trend",
                                     line=dict(color=COLORS[label], width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(xaxis_title="Prompt Tokens", yaxis_title="E2E Latency (ms)",
                      height=400, title="Prompt Length → Latency (OLS trend)",
                      legend=dict(x=.01, y=.99))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
Steep trend line = latency is driven by prompt length (prefill-bound).
Flat trend line = prompt length matters less; decode or queue time dominates.
""")

    # ── Output tokens vs E2E ──────────────────────────────────────────────────
    st.subheader("Output Length vs E2E Latency")
    fig = go.Figure()
    for label in MODELS:
        s = req_df[req_df["model_label"] == label]
        fig.add_trace(go.Scatter(
            x=s["output_tokens"], y=s["total_latency_ms"], mode="markers", name=label,
            marker=dict(color=COLORS[label], opacity=.5, size=7),
            customdata=list(zip(s["prompt_id"], s["prompt_tokens"])),
            hovertemplate="<b>" + label + "</b><br>Output: %{x} tok<br>E2E: %{y:.0f} ms<br>"
                          "Prompt: %{customdata[1]} tok<br>ID: %{customdata[0]}<extra></extra>"))
        xf, yf = _ols(s["output_tokens"], s["total_latency_ms"])
        if len(xf):
            slope = np.polyfit(s["output_tokens"].dropna().values, s["total_latency_ms"].dropna().values, 1)[0]
            fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines",
                                     name=f"{label} trend ({slope:.1f} ms/tok)",
                                     line=dict(color=COLORS[label], width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(xaxis_title="Output Tokens", yaxis_title="E2E Latency (ms)",
                      height=400, title="Output Length → Latency (slope ≈ ms/token)",
                      legend=dict(x=.01, y=.99))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
The trend-line slope approximates **ms per output token** — useful for SLA planning.
Compare with the TPOT in the Summary table; they should broadly agree.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Sanity Checks
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Timing Consistency Checks")
    st.markdown(
        "These checks verify that the request-level fields are internally consistent: "
        "`ttft + tpot × (output_tokens − 1) ≈ total_latency_ms`."
    )

    check_df = req_df.copy()
    check_df["computed_e2e_ms"] = check_df["ttft_ms"] + check_df["tpot_ms"] * (check_df["output_tokens"] - 1).clip(lower=0)
    check_df["residual_ms"]    = check_df["total_latency_ms"] - check_df["computed_e2e_ms"]
    check_df["residual_pct"]   = check_df["residual_ms"] / check_df["total_latency_ms"] * 100

    # ── Per-model residual stats ──────────────────────────────────────────────
    for label in MODELS:
        c = check_df[check_df["model_label"] == label]
        res = c["residual_ms"]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric(f"{label} — Mean |Residual|", f"{res.abs().mean():.3f} ms")
        col_b.metric("Max |Residual|", f"{res.abs().max():.3f} ms")
        col_c.metric("Mean Residual %", f"{c['residual_pct'].mean():.4f} %")
        col_d.metric("Requests", len(c))

    # ── Scatter: computed vs actual E2E ───────────────────────────────────────
    st.subheader("Computed vs Actual E2E Latency")
    fig = go.Figure()
    rng = [check_df["total_latency_ms"].min() * .95, check_df["total_latency_ms"].max() * 1.05]
    fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines", name="y = x (perfect)",
                             line=dict(color="grey", dash="dash", width=1)))
    for label in MODELS:
        c = check_df[check_df["model_label"] == label]
        fig.add_trace(go.Scatter(
            x=c["total_latency_ms"], y=c["computed_e2e_ms"],
            mode="markers", name=label,
            marker=dict(color=COLORS[label], size=6, opacity=.6),
            customdata=list(zip(c["prompt_id"], c["residual_ms"].round(3))),
            hovertemplate="<b>" + label + "</b><br>Actual: %{x:.1f} ms<br>Computed: %{y:.1f} ms<br>"
                          "Residual: %{customdata[1]} ms<br>ID: %{customdata[0]}<extra></extra>"))
    fig.update_layout(xaxis_title="Actual E2E (ms)", yaxis_title="TTFT + TPOT×(N−1) (ms)",
                      height=440, title="Do TTFT and TPOT Add Up to E2E?",
                      legend=dict(x=.01, y=.99))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
Every dot should land on the **dashed y = x** line if `total_latency_ms = ttft_ms + tpot_ms × (output_tokens − 1)`.
- Points far from the line would indicate clock skew, dropped tokens, or instrumentation bugs.
- Here the residuals are effectively zero — the streaming instrumentation is consistent.
""")

    # ── Residual histogram ────────────────────────────────────────────────────
    st.subheader("Residual Distribution")
    fig = px.histogram(check_df, x="residual_ms", color="model_label", barmode="overlay",
                       nbins=60, opacity=.65, color_discrete_map=COLORS,
                       labels={"residual_ms": "Residual (ms)", "model_label": "Model"},
                       height=300, title="Residual = Actual E2E − (TTFT + TPOT × (N−1))")
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    _note("""
A tight distribution clustered at 0 ms confirms the timing fields are self-consistent.
Any significant offset would suggest the instrumentation is losing time somewhere.
""")

    # ── GPU-time vs request-level cross-check ─────────────────────────────────
    st.subheader("GPU Fwd-Pass Time vs Request-Level Decode Time")
    st.markdown(
        "Cross-check: for each request, compare the sum of all decode fwd-pass durations "
        "(GPU time) against the request-level `tpot × (N−1)` (wall-clock decode time)."
    )
    xrows = []
    for label in MODELS:
        jdf = join_df[join_df["model_label"] == label].sort_values("rel_start_s")
        grp = jdf.groupby("request_id", group_keys=False)
        decode_gpu = grp["duration_ms"].apply(lambda s: s.iloc[1:].sum()).reset_index()
        decode_gpu.columns = ["request_id", "decode_gpu_ms"]
        rm = req_df[req_df["model_label"] == label][["request_id", "decode_ms", "model_label"]].copy()
        merged = rm.merge(decode_gpu, on="request_id", how="inner")
        xrows.append(merged)
    xdf = pd.concat(xrows, ignore_index=True)

    fig = go.Figure()
    rngs = [0, max(xdf["decode_ms"].max(), xdf["decode_gpu_ms"].max()) * 1.05]
    fig.add_trace(go.Scatter(x=rngs, y=rngs, mode="lines", name="y = x",
                             line=dict(color="grey", dash="dash", width=1)))
    for label in MODELS:
        c = xdf[xdf["model_label"] == label]
        fig.add_trace(go.Scatter(
            x=c["decode_ms"], y=c["decode_gpu_ms"],
            mode="markers", name=label,
            marker=dict(color=COLORS[label], size=6, opacity=.55),
            hovertemplate="<b>" + label + "</b><br>Request decode: %{x:.1f} ms<br>"
                          "GPU fwd sum: %{y:.1f} ms<extra></extra>"))
    fig.update_layout(xaxis_title="Request-Level Decode (tpot × (N−1)) ms",
                      yaxis_title="Sum of Decode Fwd-Pass Durations (ms)",
                      height=440,
                      title="Wall-Clock Decode vs GPU Decode — gap = inter-pass scheduling overhead",
                      legend=dict(x=.01, y=.99))
    st.plotly_chart(fig, use_container_width=True)
    _note("""
Points **below the y = x line** are expected: GPU time < wall-clock time because
there is scheduler overhead *between* decode passes (request selection, memory management, etc.).
The vertical gap is the aggregate inter-pass scheduling cost per request.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Export / Compare Tab
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ec:
    st.subheader("Export Traces as Perfetto JSON")
    st.markdown(
        "Download trace data in [Chrome Trace Event Format](https://chromium.googlesource.com/catapult/+/HEAD/docs/trace-event-format.md) "
        "— open directly in [Perfetto UI](https://ui.perfetto.dev/) for detailed inspection."
    )

    if not has_per_token:
        st.warning("Switch to **Streaming** mode for per-token data in the export.")

    # ── Export section ────────────────────────────────────────────────────
    exp_choice = st.radio("Export scope", ["Both models (combined)", *list(MODELS.keys())],
                          horizontal=True, key="exp_scope")

    if exp_choice == "Both models (combined)":
        labels_to_export = list(MODELS.keys())
    else:
        labels_to_export = [exp_choice]

    all_events = []
    pid_offset = 0
    for lbl in labels_to_export:
        evts = _build_perfetto_json(lbl, req_df, join_df, tok_df)
        if len(labels_to_export) > 1 and pid_offset > 0:
            max_pid = max((e.get("pid", 0) for e in evts), default=0)
            for e in evts:
                if "pid" in e:
                    e["pid"] += pid_offset
            pid_offset += max_pid + 1
        else:
            if evts:
                pid_offset = max((e.get("pid", 0) for e in evts), default=0) + 1
        all_events.extend(evts)

    n_reqs   = len(set(e.get("pid") for e in all_events if e.get("ph") != "M"))
    n_phases = sum(1 for e in all_events if e.get("cat") == "phase")
    n_fwd    = sum(1 for e in all_events if e.get("cat") == "fwd")
    n_tok    = sum(1 for e in all_events if e.get("cat") == "token")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Requests", n_reqs)
    mc2.metric("Phase events", n_phases)
    mc3.metric("Fwd pass events", n_fwd)
    mc4.metric("Token events", n_tok)

    perfetto_payload = json.dumps({"traceEvents": all_events}, separators=(",", ":"))
    fname = "vllm_traces_" + "_".join(l.replace(" ", "_") for l in labels_to_export) + ".json"
    st.download_button(
        "Download Perfetto JSON",
        data=perfetto_payload.encode(),
        file_name=fname,
        mime="application/json",
    )

    st.divider()

    # ── Compare section ───────────────────────────────────────────────────
    st.subheader("Compare: Upload a Perfetto JSON")
    st.markdown(
        "Upload a Perfetto JSON (e.g. from a MIST run or another vLLM config) to compare "
        "against the current traces side by side."
    )

    uploaded = st.file_uploader("Upload Perfetto JSON", type=["json"], key="perf_upload")
    if uploaded is not None:
        try:
            uploaded_data = json.loads(uploaded.read())
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid Perfetto trace JSON.")
            uploaded_data = None

        if uploaded_data is not None:
            uploaded_reqs = _parse_uploaded_perfetto(uploaded_data)
            if not uploaded_reqs:
                st.warning("No request data found in the uploaded trace. "
                           "Expected Chrome Trace Event Format with phase/token events.")
            else:
                compare_model = st.selectbox("Compare against", list(MODELS.keys()), key="cmp_model")

                # Build baseline metrics from current traces
                base_reqs = req_df[req_df["model_label"] == compare_model].copy()
                base_decode_itls = []
                if tok_df is not None:
                    for _, br in base_reqs.iterrows():
                        rt = tok_df[(tok_df["model_label"] == compare_model) &
                                    (tok_df["request_id"] == br["request_id"])]
                        itls = rt[rt["token_idx"] > 0]["itl_ms"].tolist()
                        base_decode_itls.extend(itls)

                udf = pd.DataFrame(uploaded_reqs)

                # Summary comparison
                st.markdown("#### Aggregate Comparison")
                cmp_data = {
                    "Metric": ["Requests", "Median TTFT (ms)", "Median E2E (ms)",
                               "Median ITL (ms)", "p95 ITL (ms)", "Avg Tokens"],
                    f"{compare_model} (baseline)": [
                        len(base_reqs),
                        f"{base_reqs['ttft_ms'].median():.1f}" if not base_reqs.empty else "–",
                        f"{base_reqs['total_latency_ms'].median():.1f}" if not base_reqs.empty else "–",
                        f"{np.median(base_decode_itls):.1f}" if base_decode_itls else "–",
                        f"{np.percentile(base_decode_itls, 95):.1f}" if base_decode_itls else "–",
                        f"{base_reqs['output_tokens'].mean():.0f}" if not base_reqs.empty else "–",
                    ],
                    "Uploaded trace": [
                        len(udf),
                        f"{udf['ttft_ms'].median():.1f}" if not udf.empty else "–",
                        f"{udf['e2e_ms'].median():.1f}" if not udf.empty else "–",
                        f"{udf['median_itl_ms'].median():.1f}" if not udf.empty else "–",
                        f"{udf['p95_itl_ms'].median():.1f}" if not udf.empty else "–",
                        f"{udf['tokens'].mean():.0f}" if not udf.empty else "–",
                    ],
                }
                st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

                # Per-request comparison chart: E2E latency
                st.markdown("#### E2E Latency Distribution")
                fig_cmp = go.Figure()
                if not base_reqs.empty:
                    fig_cmp.add_trace(go.Histogram(
                        x=base_reqs["total_latency_ms"], name=f"{compare_model} (baseline)",
                        opacity=0.7, marker_color=COLORS[compare_model]))
                if not udf.empty:
                    fig_cmp.add_trace(go.Histogram(
                        x=udf["e2e_ms"], name="Uploaded",
                        opacity=0.7, marker_color="#17BECF"))
                fig_cmp.update_layout(
                    barmode="overlay", xaxis_title="E2E Latency (ms)",
                    yaxis_title="Count", height=350,
                    margin=dict(l=60, r=20, t=30, b=50))
                st.plotly_chart(fig_cmp, use_container_width=True)

                # ITL distribution comparison
                if base_decode_itls and not udf.empty:
                    all_uploaded_itls = []
                    for itls in udf["itl_list"]:
                        all_uploaded_itls.extend(itls)

                    if all_uploaded_itls:
                        st.markdown("#### Inter-Token Latency Distribution")
                        fig_itl = go.Figure()
                        fig_itl.add_trace(go.Histogram(
                            x=base_decode_itls, name=f"{compare_model} (baseline)",
                            opacity=0.7, marker_color=COLORS[compare_model]))
                        fig_itl.add_trace(go.Histogram(
                            x=all_uploaded_itls, name="Uploaded",
                            opacity=0.7, marker_color="#17BECF"))
                        fig_itl.update_layout(
                            barmode="overlay", xaxis_title="ITL (ms)",
                            yaxis_title="Count", height=350,
                            margin=dict(l=60, r=20, t=30, b=50))
                        st.plotly_chart(fig_itl, use_container_width=True)

                # TTFT distribution comparison
                st.markdown("#### TTFT Distribution")
                fig_ttft = go.Figure()
                if not base_reqs.empty:
                    fig_ttft.add_trace(go.Histogram(
                        x=base_reqs["ttft_ms"], name=f"{compare_model} (baseline)",
                        opacity=0.7, marker_color=COLORS[compare_model]))
                if not udf.empty:
                    fig_ttft.add_trace(go.Histogram(
                        x=udf["ttft_ms"], name="Uploaded",
                        opacity=0.7, marker_color="#17BECF"))
                fig_ttft.update_layout(
                    barmode="overlay", xaxis_title="TTFT (ms)",
                    yaxis_title="Count", height=350,
                    margin=dict(l=60, r=20, t=30, b=50))
                st.plotly_chart(fig_ttft, use_container_width=True)

                _note("""
**Comparison guide:**

- **Aggregate table** — median metrics across all requests. Lower is better for latency metrics.
- **E2E Latency** — overlapping histograms show the full distribution. If the uploaded trace is shifted left, it's faster overall.
- **ITL** — inter-token latency governs perceived streaming smoothness. Tighter, left-shifted = better.
- **TTFT** — time-to-first-token affects perceived responsiveness. Lower = snappier first response.

Re-export a trace from a different configuration (e.g. MIST) and upload it here to compare.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab — Fwd Pass Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fwd_cmp:
    st.subheader("Forward Pass Comparison")
    st.markdown("""
Compare two forward-pass traces side by side.  Useful for comparing different
runs of the same model/hw/input (vLLM vs vLLM, vLLM vs MIST, vLLM vs SGLang).
Forward passes exceeding the difference threshold are highlighted.
""")

    # ── Helper: load a fwd JSONL into a DataFrame ────────────────────────────
    def _load_fwd_jsonl(source) -> pd.DataFrame:
        """Load fwd JSONL from a file path (str/Path) or uploaded file object."""
        rows = []
        if isinstance(source, (str, Path)):
            with open(source) as f:
                for line in f:
                    rows.append(json.loads(line))
        else:
            for line in source:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        if "batch_size" not in df.columns and "req_ids" in df.columns:
            df["batch_size"] = df["req_ids"].apply(len)
        return df

    # ── Discover existing fwd traces ─────────────────────────────────────────
    existing_fwd_files = {}
    for d in [TRACES_DIR, TRACES_DIR / "per-token", TRACES_DIR / "streaming", TRACES_DIR / "multiturn"]:
        if d.is_dir():
            for p in sorted(d.glob("*_fwd.jsonl")):
                existing_fwd_files[f"{p.parent.name}/{p.name}" if p.parent != TRACES_DIR else p.name] = p

    # ── File selection ───────────────────────────────────────────────────────
    src_col1, src_col2 = st.columns(2)
    df_a, df_b = None, None
    label_a, label_b = "Run A", "Run B"

    with src_col1:
        st.markdown("**Run A**")
        src_a = st.radio("Source A", ["Existing trace", "Upload file"], key="src_a", horizontal=True)
        if src_a == "Existing trace" and existing_fwd_files:
            sel_a = st.selectbox("Select trace A", list(existing_fwd_files.keys()), key="sel_a")
            if sel_a:
                df_a = _load_fwd_jsonl(existing_fwd_files[sel_a])
                label_a = sel_a
        else:
            up_a = st.file_uploader("Upload fwd JSONL (A)", type=["jsonl"], key="up_a")
            if up_a is not None:
                df_a = _load_fwd_jsonl(up_a)
                label_a = up_a.name
        label_a = st.text_input("Label A", value=label_a, key="lbl_a")

    with src_col2:
        st.markdown("**Run B**")
        src_b = st.radio("Source B", ["Existing trace", "Upload file"], key="src_b", horizontal=True)
        if src_b == "Existing trace" and existing_fwd_files:
            sel_b = st.selectbox("Select trace B", list(existing_fwd_files.keys()), key="sel_b",
                                 index=min(1, len(existing_fwd_files) - 1))
            if sel_b:
                df_b = _load_fwd_jsonl(existing_fwd_files[sel_b])
                label_b = sel_b
        else:
            up_b = st.file_uploader("Upload fwd JSONL (B)", type=["jsonl"], key="up_b")
            if up_b is not None:
                df_b = _load_fwd_jsonl(up_b)
                label_b = up_b.name
        label_b = st.text_input("Label B", value=label_b, key="lbl_b")

    if df_a is not None and df_b is not None:
        # ── Config ───────────────────────────────────────────────────────────
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            threshold = st.slider("Difference threshold (ms)", 0.0, 20.0, 2.0, 0.5, key="cmp_thresh")
        with cfg_col2:
            align_method = st.radio("Alignment method",
                                    ["By fwd_id (index)", "By batch composition (Jaccard)"],
                                    key="cmp_align", horizontal=True)

        # ── Alignment ────────────────────────────────────────────────────────
        if align_method == "By fwd_id (index)":
            n = min(len(df_a), len(df_b))
            if len(df_a) != len(df_b):
                st.warning(f"Different fwd pass counts: {label_a}={len(df_a)}, {label_b}={len(df_b)}. "
                           f"Truncating to first {n} passes.")
            merged = pd.DataFrame({
                "idx": range(n),
                "fwd_id_a": df_a["fwd_id"].iloc[:n].values,
                "fwd_id_b": df_b["fwd_id"].iloc[:n].values,
                "duration_a": df_a["duration_ms"].iloc[:n].values,
                "duration_b": df_b["duration_ms"].iloc[:n].values,
                "batch_size_a": df_a["batch_size"].iloc[:n].values,
                "batch_size_b": df_b["batch_size"].iloc[:n].values,
            })
        else:
            # Jaccard-based alignment: match each fwd pass in A to best match in B
            def _jaccard(s1, s2):
                s1, s2 = set(s1), set(s2)
                inter = len(s1 & s2)
                union = len(s1 | s2)
                return inter / union if union > 0 else 0.0

            matches = []
            used_b = set()
            for i, row_a in df_a.iterrows():
                best_j, best_score = -1, -1.0
                ids_a = row_a.get("req_ids", [])
                for j, row_b in df_b.iterrows():
                    if j in used_b:
                        continue
                    score = _jaccard(ids_a, row_b.get("req_ids", []))
                    if score > best_score:
                        best_score = score
                        best_j = j
                if best_j >= 0 and best_score > 0:
                    used_b.add(best_j)
                    matches.append((i, best_j, best_score))

            if not matches:
                st.error("No matching forward passes found by batch composition. "
                         "The traces may use different request IDs.")
                st.stop()

            merged = pd.DataFrame({
                "idx": range(len(matches)),
                "fwd_id_a": [df_a.loc[m[0], "fwd_id"] for m in matches],
                "fwd_id_b": [df_b.loc[m[1], "fwd_id"] for m in matches],
                "duration_a": [df_a.loc[m[0], "duration_ms"] for m in matches],
                "duration_b": [df_b.loc[m[1], "duration_ms"] for m in matches],
                "batch_size_a": [df_a.loc[m[0], "batch_size"] for m in matches],
                "batch_size_b": [df_b.loc[m[1], "batch_size"] for m in matches],
                "jaccard": [m[2] for m in matches],
            })
            st.info(f"Matched {len(matches)} forward passes by batch composition. "
                    f"Mean Jaccard similarity: {merged['jaccard'].mean():.2f}")

        merged["diff_ms"] = merged["duration_a"] - merged["duration_b"]
        merged["abs_diff_ms"] = merged["diff_ms"].abs()
        merged["exceeds"] = merged["abs_diff_ms"] > threshold

        # ── Summary metrics ──────────────────────────────────────────────────
        n_exceed = merged["exceeds"].sum()
        pct_exceed = 100 * n_exceed / len(merged) if len(merged) > 0 else 0
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Matched Passes", len(merged))
        m2.metric("Exceeding Threshold", f"{n_exceed} ({pct_exceed:.1f}%)")
        m3.metric("Mean |diff|", f"{merged['abs_diff_ms'].mean():.2f} ms")
        m4.metric("Max |diff|", f"{merged['abs_diff_ms'].max():.2f} ms")
        m5.metric("Median |diff|", f"{merged['abs_diff_ms'].median():.2f} ms")

        # ── Dual bar chart ───────────────────────────────────────────────────
        st.subheader("Duration Comparison")
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=merged["idx"], y=merged["duration_a"],
            name=label_a, marker_color="#636EFA",
        ))
        fig_bars.add_trace(go.Bar(
            x=merged["idx"], y=merged["duration_b"],
            name=label_b, marker_color="#EF553B",
        ))
        # Highlight exceeding passes
        for _, row in merged[merged["exceeds"]].iterrows():
            fig_bars.add_vrect(
                x0=row["idx"] - 0.5, x1=row["idx"] + 0.5,
                fillcolor="red", opacity=0.08, line_width=0,
            )
        fig_bars.update_layout(
            barmode="group", xaxis_title="Forward Pass Index",
            yaxis_title="Duration (ms)", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        # ── Difference timeline ──────────────────────────────────────────────
        st.subheader("Difference Timeline")
        fig_diff = go.Figure()
        # Threshold band
        fig_diff.add_hrect(y0=-threshold, y1=threshold,
                           fillcolor="green", opacity=0.1, line_width=0)
        fig_diff.add_hline(y=threshold, line_dash="dash", line_color="green",
                           opacity=0.5, annotation_text=f"+{threshold}ms")
        fig_diff.add_hline(y=-threshold, line_dash="dash", line_color="green",
                           opacity=0.5, annotation_text=f"-{threshold}ms")
        fig_diff.add_hline(y=0, line_color="gray", opacity=0.3)
        # Points
        colors = ["red" if e else "#636EFA" for e in merged["exceeds"]]
        fig_diff.add_trace(go.Scatter(
            x=merged["idx"], y=merged["diff_ms"],
            mode="markers", marker=dict(color=colors, size=5),
            name="diff (A - B)",
            hovertemplate="idx=%{x}<br>diff=%{y:.2f}ms<extra></extra>",
        ))
        fig_diff.update_layout(
            xaxis_title="Forward Pass Index",
            yaxis_title=f"Diff: {label_a} - {label_b} (ms)",
            height=350,
        )
        st.plotly_chart(fig_diff, use_container_width=True)

        # ── Detail table ─────────────────────────────────────────────────────
        st.subheader("Detailed Comparison")
        show_cols = ["idx", "fwd_id_a", "fwd_id_b", "duration_a", "duration_b",
                     "diff_ms", "abs_diff_ms", "batch_size_a", "batch_size_b", "exceeds"]
        if "jaccard" in merged.columns:
            show_cols.append("jaccard")

        def _highlight_exceeds(row):
            if row["exceeds"]:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        styled = (merged[show_cols]
                   .sort_values("abs_diff_ms", ascending=False)
                   .style.apply(_highlight_exceeds, axis=1)
                   .format({
                       "duration_a": "{:.2f}", "duration_b": "{:.2f}",
                       "diff_ms": "{:.2f}", "abs_diff_ms": "{:.2f}",
                       **({} if "jaccard" not in merged.columns else {"jaccard": "{:.2f}"}),
                   }))
        st.dataframe(styled, use_container_width=True, height=400)

    elif df_a is None and df_b is None:
        st.info("Select or upload two forward-pass trace files to compare.")
    else:
        st.warning("Please provide both Run A and Run B traces.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Raw Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Request Traces")
    model_filter = st.multiselect("Filter by model", list(MODELS.keys()), default=list(MODELS.keys()))
    fr = req_df[req_df["model_label"].isin(model_filter)]
    st.dataframe(
        fr[["model_label", "prompt_id", "request_id", "prompt_tokens", "output_tokens",
            "ttft_ms", "tpot_ms", "total_latency_ms", "scheduling_overhead_ms"]]
        .sort_values(["model_label", "prompt_id"]),
        use_container_width=True, height=350)
    st.download_button("Download Request Traces (CSV)", fr.to_csv(index=False).encode(),
                       file_name="request_traces.csv", mime="text/csv")

    st.subheader("Forward Pass Traces")
    ff = fwd_df[fwd_df["model_label"].isin(model_filter)]
    st.dataframe(
        ff[["model_label", "fwd_id", "rel_start_s", "rel_end_s",
            "duration_ms", "batch_size", "total_tokens"]]
        .sort_values(["model_label", "fwd_id"]),
        use_container_width=True, height=350)
    st.download_button("Download Fwd Pass Traces (CSV)",
                       ff.drop(columns=["req_ids", "num_tokens"], errors="ignore")
                       .to_csv(index=False).encode(),
                       file_name="fwd_traces.csv", mime="text/csv")
