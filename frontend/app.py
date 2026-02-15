"""
Streamlit dashboard for the AI Gateway.

Tabs:
  1. Router   — prompt input, classify, route, view response + reasoning
  2. History  — table of past routing decisions
  3. Analytics — cost savings, model distribution charts
"""

import traceback

import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Gateway",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_call(method: str, endpoint: str, **kwargs) -> dict | list | None:
    """Make an API call to the backend and return JSON, or None on error."""
    try:
        url = f"{API_BASE}{endpoint}"
        resp = requests.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure the API is running on port 8000.")
        traceback.print_exc()
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        traceback.print_exc()
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return None


def complexity_color(score: int) -> str:
    """Return a color for the complexity score."""
    try:
        if score <= 3:
            return "#2ecc71"   # green
        elif score <= 6:
            return "#f39c12"   # orange
        else:
            return "#e74c3c"   # red
    except Exception:
        traceback.print_exc()
        return "#95a5a6"


def tier_label(score: int) -> str:
    """Return a human-readable tier label."""
    try:
        if score <= 3:
            return "LOW"
        elif score <= 6:
            return "MEDIUM"
        else:
            return "HIGH"
    except Exception:
        traceback.print_exc()
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AI Gateway")
    st.caption("Intelligent LLM Router")
    st.divider()

    # Health check
    health = api_call("GET", "/health")
    if health:
        st.metric("Models Available", health["models_available"])
        st.metric("Requests Logged", health["total_requests_logged"])
        cache = health["cache_stats"]
        st.metric("Cache Hit Rate", f"{cache['hit_rate_percent']}%")
        st.caption(f"Cache: {cache['size']}/{cache['max_size']} entries")
    else:
        st.warning("Backend offline")

    st.divider()

    # Models info
    models = api_call("GET", "/models")
    if models:
        st.subheader("Available Models")
        for m in models:
            with st.expander(f"{m['name']}"):
                st.caption(f"Provider: **{m['provider']}**")
                st.caption(f"Input: ${m['cost_per_1k_input_tokens']:.4f}/1k tokens")
                st.caption(f"Output: ${m['cost_per_1k_output_tokens']:.4f}/1k tokens")
                st.caption(f"Avg latency: {m['avg_latency_ms']}ms")
                st.caption(f"Strengths: {', '.join(m['strengths'])}")


# ---------------------------------------------------------------------------
# Main content — Tabs
# ---------------------------------------------------------------------------

tab_router, tab_history, tab_analytics, tab_how_it_works = st.tabs(["Router", "History", "Analytics", "How It Works"])

# ============================= ROUTER TAB ================================

with tab_router:
    st.header("Route a Prompt")

    col_input, col_config = st.columns([3, 1])

    with col_input:
        prompt = st.text_area(
            "Enter your prompt",
            height=120,
            placeholder="e.g. Write a Python web scraper with error handling...",
        )

    with col_config:
        classifier_mode = st.radio(
            "Classifier Mode",
            options=["rule_based", "llm_based"],
            format_func=lambda x: "Rule-Based" if x == "rule_based" else "LLM-Based",
            help="Rule-based uses heuristics (fast, free). LLM-based uses Claude Haiku (slower, more accurate).",
        )

    route_clicked = st.button("Route Prompt", type="primary", use_container_width=True, disabled=not prompt.strip())

    if route_clicked and prompt.strip():
        with st.spinner("Classifying, routing, and generating response..."):
            result = api_call(
                "POST", "/route",
                json={"prompt": prompt.strip(), "classifier_mode": classifier_mode},
            )

        if result:
            st.divider()

            # --- Row 1: Classification + Routing summary ---
            c1, c2, c3, c4 = st.columns(4)

            score = result["classification"]["complexity_score"]
            task_type = result["classification"]["task_type"]
            model = result["routing"]["model"]
            provider = result["routing"]["provider"]

            with c1:
                color = complexity_color(score)
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:10px; border-radius:10px;
                                border: 2px solid {color}; background: {color}15;">
                        <div style="font-size:36px; font-weight:bold; color:{color};">{score}/10</div>
                        <div style="font-size:14px; color:{color};">{tier_label(score)} complexity</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:10px; border-radius:10px;
                                border: 2px solid #3498db; background: #3498db15;">
                        <div style="font-size:20px; font-weight:bold; color:#3498db;">{task_type.replace('_', ' ').title()}</div>
                        <div style="font-size:14px; color:#3498db;">Task Type</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:10px; border-radius:10px;
                                border: 2px solid #9b59b6; background: #9b59b615;">
                        <div style="font-size:20px; font-weight:bold; color:#9b59b6;">{model}</div>
                        <div style="font-size:14px; color:#9b59b6;">{provider}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c4:
                savings = result["cost_comparison"]["savings_percent"]
                savings_color = "#2ecc71" if savings > 0 else "#95a5a6"
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:10px; border-radius:10px;
                                border: 2px solid {savings_color}; background: {savings_color}15;">
                        <div style="font-size:36px; font-weight:bold; color:{savings_color};">{savings}%</div>
                        <div style="font-size:14px; color:{savings_color};">Cost Saved</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()

            # --- Row 2: Response + Details side by side ---
            col_resp, col_details = st.columns([3, 2])

            with col_resp:
                st.subheader("Model Response")
                st.markdown(result["response"]["response_text"])

                # Response metadata
                resp = result["response"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Tokens Used", resp["tokens_used"])
                m2.metric("Latency", f"{resp['latency_ms']}ms")
                m3.metric("Cost", f"${resp['simulated_cost']:.6f}")

            with col_details:
                # Reasoning chain
                st.subheader("Routing Reasoning")
                for step in result["routing"]["reasoning_chain"]:
                    st.markdown(f"**Step {step['step']}:** {step['description']}")

                st.divider()

                # Cost comparison detail
                st.subheader("Cost Comparison")
                cc = result["cost_comparison"]
                cost_df = pd.DataFrame({
                    "Model": [cc["chosen_model"], cc["baseline_model"]],
                    "Cost ($)": [cc["chosen_cost"], cc["baseline_cost"]],
                })
                fig_cost = px.bar(
                    cost_df, x="Model", y="Cost ($)",
                    color="Model",
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                    title="Cost: Chosen vs Baseline (GPT-4o)",
                )
                fig_cost.update_layout(showlegend=False, height=280)
                st.plotly_chart(fig_cost, use_container_width=True)

                # Classification reasoning
                st.subheader("Classification Details")
                st.info(result["classification"]["reasoning"])
                conf = result["classification"]["confidence"]
                st.progress(conf, text=f"Confidence: {conf:.0%}")


# ============================= HISTORY TAB ================================

with tab_history:
    st.header("Routing History")

    logs = api_call("GET", "/logs?limit=100")

    if logs is None:
        st.info("Backend not available.")
    elif len(logs) == 0:
        st.info("No requests logged yet. Use the Router tab to make some requests!")
    else:
        try:
            df = pd.DataFrame(logs)
            df = df.rename(columns={
                "request_id": "Request ID",
                "timestamp": "Timestamp",
                "prompt_snippet": "Prompt",
                "classifier_mode": "Classifier",
                "complexity_score": "Score",
                "task_type": "Task Type",
                "routed_model": "Model",
                "latency_ms": "Latency (ms)",
                "cost": "Cost ($)",
            })

            # Color code by complexity
            st.dataframe(
                df[["Timestamp", "Prompt", "Classifier", "Score", "Task Type", "Model", "Latency (ms)", "Cost ($)"]],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Error displaying logs: {e}")
            traceback.print_exc()


# ============================= ANALYTICS TAB ================================

with tab_analytics:
    st.header("Gateway Analytics")

    stats = api_call("GET", "/stats")

    if stats is None:
        st.info("Backend not available.")
    elif stats["total_requests"] == 0:
        st.info("No data yet. Use the Router tab to make some requests!")
    else:
        try:
            # --- Top-level metrics ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Requests", stats["total_requests"])
            k2.metric("Total Cost", f"${stats['total_cost']:.4f}")
            k3.metric("Total Savings", f"${stats['total_savings']:.4f}")
            k4.metric("Savings Rate", f"{stats['savings_percent']}%")

            st.divider()

            col_pie, col_bar = st.columns(2)

            model_usage = stats["model_usage"]
            if model_usage:
                usage_df = pd.DataFrame(model_usage)

                # Pie chart — model distribution by request count
                with col_pie:
                    st.subheader("Model Distribution")
                    fig_pie = px.pie(
                        usage_df,
                        values="request_count",
                        names="model",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hole=0.4,
                    )
                    fig_pie.update_layout(height=380)
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Bar chart — cost per model
                with col_bar:
                    st.subheader("Cost by Model")
                    fig_bar = px.bar(
                        usage_df,
                        x="model",
                        y="total_cost",
                        color="model",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={"total_cost": "Total Cost ($)", "model": "Model"},
                    )
                    fig_bar.update_layout(showlegend=False, height=380)
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.divider()

                col_lat, col_summary = st.columns(2)

                # Latency comparison
                with col_lat:
                    st.subheader("Average Latency by Model")
                    fig_lat = px.bar(
                        usage_df,
                        x="model",
                        y="avg_latency_ms",
                        color="model",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        labels={"avg_latency_ms": "Avg Latency (ms)", "model": "Model"},
                    )
                    fig_lat.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_lat, use_container_width=True)

                # Savings gauge
                with col_summary:
                    st.subheader("Cost Efficiency")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=stats["savings_percent"],
                        number={"suffix": "%"},
                        title={"text": "Overall Savings vs GPT-4o Baseline"},
                        delta={"reference": 50, "increasing": {"color": "#2ecc71"}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#2ecc71"},
                            "steps": [
                                {"range": [0, 30], "color": "#fadbd8"},
                                {"range": [30, 70], "color": "#fdebd0"},
                                {"range": [70, 100], "color": "#d5f5e3"},
                            ],
                        },
                    ))
                    fig_gauge.update_layout(height=350)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.metric("Avg Prompt Complexity", f"{stats['avg_complexity']}/10")

                    # Baseline vs actual cost comparison
                    st.markdown(
                        f"**If all requests used GPT-4o:** ${stats['total_baseline_cost']:.4f}  \n"
                        f"**Actual cost with routing:** ${stats['total_cost']:.4f}  \n"
                        f"**You saved:** ${stats['total_savings']:.4f} ({stats['savings_percent']}%)"
                    )

        except Exception as e:
            st.error(f"Error rendering analytics: {e}")
            traceback.print_exc()


# ============================= HOW IT WORKS TAB ================================

with tab_how_it_works:
    st.header("How the AI Gateway Works")

    st.markdown("""
    The AI Gateway intelligently routes every prompt through a **3-stage pipeline**:
    **Classify** the prompt's complexity, **Route** it to the optimal model, and **Generate**
    a response — all while tracking cost savings.
    """)

    st.divider()

    # --- Architecture overview ---
    st.subheader("Architecture Overview")

    st.code("""
    User Prompt
         │
         ▼
    ┌─────────────────┐
    │   Classifier     │  ← Rule-Based (heuristics) OR LLM-Based (Claude Haiku)
    │   Output:        │
    │   - Score (1-10) │
    │   - Task Type    │
    │   - Reasoning    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Router         │  ← Maps (complexity tier + task type) → best model
    │   Output:        │
    │   - Model choice │
    │   - 5-step chain │
    │   - Cost estimate│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Provider       │  ← Real API call to OpenAI or Anthropic
    │   Output:        │
    │   - Response text│
    │   - Token count  │
    │   - Actual cost  │
    └─────────────────┘
    """, language=None)

    st.divider()

    # --- Classifier modes ---
    st.subheader("Classifier Modes")

    col_rule, col_llm = st.columns(2)

    with col_rule:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; border:2px solid #3498db; background:#3498db10;">
        <h3 style="color:#3498db; margin-top:0;">Rule-Based Classifier</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### How it works")
        st.markdown("""
        A **deterministic heuristic engine** that analyzes the prompt text
        using pattern matching and keyword detection. No API call needed.

        **5-Stage Pipeline:**

        **1. Token Estimation**
        - Approximates token count from word count and character count
        - Formula: `(words * 0.75 + chars / 4) / 2`

        **2. Task Type Detection**
        - Scans prompt against 7 keyword banks (one per task type)
        - Each bank has 5-25 regex patterns
        - The task type with the most pattern matches wins
        - Falls back to `general` if no strong signals
        """)

        st.markdown("**Keyword Banks:**")
        task_examples = {
            "Code": '`python`, `function`, `debug`, `api`, `build`...',
            "Math": '`solve`, `integral`, `equation`, `probability`...',
            "Creative": '`poem`, `story`, `haiku`, `compose`, `fiction`...',
            "Analysis": '`analyze`, `compare`, `pros and cons`, `evaluate`...',
            "Translation": '`translate`, `in french`, `in spanish`...',
            "Reasoning": '`step by step`, `quantum`, `philosophy`, `paradox`...',
            "Simple QA": '`what is`, `capital of`, `define`, `how many`...',
        }
        for task, examples in task_examples.items():
            st.markdown(f"- **{task}:** {examples}")

        st.markdown("""
        **3. Base Complexity Score**
        - Each task type has a base score: Simple QA = 2, Translation = 3,
          Creative/Code/Analysis = 5, Math = 6, Reasoning = 7
        - Longer prompts get +1 (>80 tokens) or +2 (>200 tokens)

        **4. Booster & Reducer Adjustments**
        """)

        boost_data = {
            "Signal": ["'step by step', 'detailed'", "'compare', 'trade-offs'",
                       "'advanced', 'complex'", "'architecture', 'system design'",
                       "'simple', 'basic', 'easy'", "'yes or no'", "Very short prompt (<30 chars)"],
            "Effect": ["+2", "+1", "+2", "+2", "-1", "-2", "-1"],
        }
        st.dataframe(pd.DataFrame(boost_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **5. Clamp to [1, 10]**
        - Final score is clamped to the valid range
        - Full reasoning string is assembled from all factors

        ---

        **Pros:**
        - Instant (no API call)
        - Free (no cost)
        - Deterministic (same prompt = same result every time)
        - Confidence is always 1.0

        **Cons:**
        - Can miss nuance and context
        - Relies on keyword presence, not semantic understanding
        - Novel phrasing may not match patterns
        """)

    with col_llm:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; border:2px solid #9b59b6; background:#9b59b610;">
        <h3 style="color:#9b59b6; margin-top:0;">LLM-Based Classifier</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### How it works")
        st.markdown("""
        Sends the prompt to a **real LLM** (Claude Haiku or GPT-4o-mini) with a
        carefully crafted system prompt. The LLM returns a structured JSON
        classification — leveraging true language understanding.

        **Pipeline:**

        **1. System Prompt**
        - Instructs the LLM to act as a "prompt complexity classifier"
        - Provides scoring guidelines (1-3: simple, 4-6: moderate, 7-10: complex)
        - Defines all 8 task types with descriptions
        - Requests raw JSON output with 4 fields

        **2. LLM Call**
        - Sends user prompt to the classifier model
        - Model: **Claude Haiku 4.5** (default) or **GPT-4o-mini**
        - Configurable via `CLASSIFIER_LLM_PROVIDER` env var
        - Max tokens: 300 (classification only, not generation)

        **3. Response Parsing**
        - Strips markdown code fences if present
        - Parses JSON and validates every field:
          - `complexity_score`: clamped to [1, 10]
          - `task_type`: falls back to `general` if invalid
          - `confidence`: clamped to [0.0, 1.0]
          - `reasoning`: uses fallback if empty

        **4. Error Handling**
        - Invalid JSON → raises clear error with raw response
        - Missing fields → uses sensible defaults
        - API failures → full traceback logging
        """)

        st.markdown("**Example LLM Response:**")
        st.code("""{
  "complexity_score": 6,
  "task_type": "code",
  "reasoning": "This requires writing functional code with error
    handling, involving HTTP requests, parsing, and exception
    management. Moderately complex.",
  "confidence": 0.92
}""", language="json")

        st.markdown("""
        ---

        **Pros:**
        - Understands context and nuance
        - Handles novel/unusual phrasing
        - Provides richer, more detailed reasoning
        - Variable confidence reflects actual certainty

        **Cons:**
        - Costs ~$0.0001 per classification call
        - Adds 200-500ms latency
        - Non-deterministic (slight variation between calls)
        """)

    st.divider()

    # --- Routing table ---
    st.subheader("Routing Table")

    st.markdown("""
    After classification, the router maps **(complexity tier + task type)** to the optimal model:
    """)

    routing_data = {
        "Complexity": ["1-3 (Low)", "1-3 (Low)", "4-6 (Medium)", "4-6 (Medium)",
                       "7-10 (High)", "7-10 (High)"],
        "Task Types": [
            "All types",
            "",
            "Code, Math, General, Simple QA",
            "Analysis, Creative, Translation, Reasoning",
            "Reasoning, Math, Code, Simple QA",
            "Analysis, Creative, General, Translation",
        ],
        "Model": [
            "GPT-4o-mini", "",
            "GPT-4o-mini", "Claude 3.5 Sonnet",
            "GPT-4o", "Claude 3.5 Sonnet",
        ],
        "Why": [
            "Fast and cheapest option for simple tasks",
            "",
            "Cost-effective for standard code and math",
            "Excels at nuanced, creative, and analytical work",
            "Top-tier reasoning for the hardest problems",
            "Best at long-form analysis and creative content",
        ],
    }
    # Remove empty rows
    routing_df = pd.DataFrame(routing_data)
    routing_df = routing_df[routing_df["Task Types"] != ""]
    st.dataframe(routing_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Model comparison ---
    st.subheader("Model Comparison")

    model_data = {
        "Model": ["GPT-4o-mini", "Claude 3.5 Sonnet", "GPT-4o"],
        "Provider": ["OpenAI", "Anthropic", "OpenAI"],
        "Input Cost ($/1k tokens)": ["$0.00015", "$0.003", "$0.005"],
        "Output Cost ($/1k tokens)": ["$0.0006", "$0.015", "$0.015"],
        "Avg Latency": ["~300ms", "~700ms", "~800ms"],
        "Best For": [
            "Simple QA, translations, basic code",
            "Analysis, creative writing, nuanced content",
            "Complex math, advanced reasoning, hard code",
        ],
    }
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    st.markdown("""
    > **Baseline model:** GPT-4o is the most expensive model. All cost savings
    > are calculated by comparing the chosen model's cost against what GPT-4o
    > would have cost for the same number of tokens.
    """)

    st.divider()

    # --- Caching ---
    st.subheader("Response Caching")

    st.markdown("""
    The gateway caches responses to avoid redundant API calls:

    - **Key:** SHA-256 hash of `(prompt text + classifier mode)`
    - **LRU Eviction:** Max 100 entries — least recently used entry is evicted when full
    - **TTL Expiry:** Entries automatically expire after 30 minutes
    - **Thread-Safe:** All operations are locked for concurrent request safety

    When a cached response is found, it's returned **instantly** (typically <10ms)
    with zero API cost. Cache stats are visible in the sidebar.
    """)
