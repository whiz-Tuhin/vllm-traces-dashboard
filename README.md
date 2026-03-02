# vLLM Trace Comparison Dashboard

Interactive Streamlit dashboard comparing vLLM inference traces for:
- **Llama-2-7b-chat-hf** (Meta)
- **Qwen2.5-7B-Instruct** (Alibaba)

Traces were collected on an L40s GPU (48 GB VRAM) using vLLM 0.7.3 with 200 ShareGPT prompts,
concurrency=16, max_tokens=256. Both **streaming** and **non-streaming** modes are included.

---

## Folder Structure

```
├── app.py              # Streamlit dashboard
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── traces/
    ├── meta-llama_Llama-2-7b-chat-hf.jsonl        # Non-streaming request traces (200)
    ├── meta-llama_Llama-2-7b-chat-hf_fwd.jsonl    # Non-streaming forward-pass traces
    ├── Qwen_Qwen2.5-7B-Instruct.jsonl
    ├── Qwen_Qwen2.5-7B-Instruct_fwd.jsonl
    └── streaming/
        ├── meta-llama_Llama-2-7b-chat-hf-streaming.jsonl        # Streaming request traces (200)
        ├── meta-llama_Llama-2-7b-chat-hf-streaming_fwd.jsonl
        ├── Qwen_Qwen2.5-7B-Instruct-streaming.jsonl
        └── Qwen_Qwen2.5-7B-Instruct-streaming_fwd.jsonl
```

---

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501**.

---

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Summary stats, E2E & TTFT CDFs, TPOT violin, fwd-pass box plot, median phase breakdown, latency histogram |
| **Forward Pass Timeline** | Every fwd pass plotted over time (log y), batch size over time with rolling avg |
| **Per-Request Deep Dive** | Select any prompt → lifecycle Gantt with forward-pass tick marks, per-pass scatter (★ prefill, ● decode) |
| **Batching & Throughput** | Batch size histogram, token throughput, prompt/output length vs latency with OLS trends |
| **Sanity Checks** | TTFT + TPOT × (N−1) vs E2E consistency, residual analysis, GPU vs request-level decode cross-check |
| **Raw Data** | Filterable tables + CSV download |

---

## Trace Schema

### Request trace (`*.jsonl`)

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | str | vLLM internal ID |
| `prompt_id` | str | Stable cross-model ID (`prompt_00042`) |
| `prompt_tokens` | int | Tokens in the prompt |
| `output_tokens` | int | Tokens generated |
| `ttft_ms` | float | Time-to-first-token (ms) |
| `tpot_ms` | float | Time-per-output-token (ms) |
| `total_latency_ms` | float | End-to-end latency (ms) |
| `scheduling_overhead_ms` | float | API receive → engine enqueue (ms) |

### Forward pass trace (`*_fwd.jsonl`)

| Field | Type | Description |
|-------|------|-------------|
| `fwd_id` | int | Sequential forward pass counter |
| `start_ts` / `end_ts` | float | Wall-clock timestamps (Unix) |
| `duration_ms` | float | CUDA-timed GPU duration (ms) |
| `req_ids` | list | Request IDs in this batch |
| `num_tokens` | dict | `{request_id: tokens_processed}` |
| `total_tokens` | int | Sum of tokens across all requests |
