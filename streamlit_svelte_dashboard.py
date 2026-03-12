#!/usr/bin/env python3
"""
Streamlit wrapper for the Svelte dashboard.

Usage:
    source venv/bin/activate
    streamlit run streamlit_svelte_dashboard.py

Optional env var:
    SVELTE_DASHBOARD_URL=http://localhost:5175
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import streamlit as st
import streamlit.components.v1 as components


def check_url(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except (urllib.error.URLError, TimeoutError):
        return False


st.set_page_config(page_title="vLLM Traces Dashboard (Svelte)", layout="wide")

default_url = os.environ.get("SVELTE_DASHBOARD_URL", "http://localhost:5175")
st.title("vLLM Traces Dashboard (Svelte)")
st.caption("Streamlit wrapper for the modern Svelte dashboard")

url = st.text_input("Dashboard URL", value=default_url)

col1, col2 = st.columns([1, 3])
with col1:
    refresh = st.button("Refresh health check")
with col2:
    if refresh or True:
        healthy = check_url(url)
        if healthy:
            st.success(f"Dashboard reachable at {url}")
        else:
            st.warning(
                "Dashboard not reachable. Start it first from `dashboard-v2/` with:\n"
                "`npm run dev -- --port 5175`"
            )

components.iframe(url, height=1000, scrolling=True)

