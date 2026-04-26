"""
Streamlit Dashboard v2 — Prompt Injection Defense Solution Framework
Pages: Overview · Detect · Run Simulation · Single Attack · Adversarial Loop ·
       Benchmark · Compare Models · Insights · Logs · Replay
"""
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Prompt Injection Defense Framework",
    page_icon="🛡️",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path, params=None, timeout=10):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error [{path}]: {e}")
        return None


def api_post(path, body):
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=600)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error [{path}]: {e}")
        return None


def tier_badge(tier):
    return {"block": "🔴 BLOCK", "warn": "🟡 WARN", "allow": "🟢 ALLOW"}.get(tier, tier)


def risk_color(score):
    if score >= 0.70:
        return "🔴"
    if score >= 0.40:
        return "🟡"
    return "🟢"


def score_gauge(score, title="Risk Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 3),
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "darkred" if score >= 0.7 else ("orange" if score >= 0.4 else "green")},
            "steps": [
                {"range": [0, 0.4], "color": "#d4edda"},
                {"range": [0.4, 0.7], "color": "#fff3cd"},
                {"range": [0.7, 1.0], "color": "#f8d7da"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


@st.cache_data(ttl=120)
def get_models():
    data = api_get("/models") or {}
    return data.get("models", ["llama3"])


@st.cache_data(ttl=300)
def get_attacks():
    return api_get("/attacks") or []


@st.cache_data(ttl=30)
def get_health():
    return api_get("/health") or {}


@st.cache_data(ttl=20)
def get_overview_data():
    return api_get("/logs/runs") or [], api_get("/insights") or {}


@st.cache_data(ttl=15)
def get_insights_data():
    return api_get("/insights") or {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ Defense Framework")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Detect (Live)",
    "Run Simulation",
    "Single Attack",
    "Adversarial Loop",
    "Benchmark",
    "Compare Models",
    "Insights",
    "Logs",
    "Replay",
])

health = get_health()
if health:
    st.sidebar.markdown(
        f"**Ollama:** {'✅ Online' if health.get('ollama_available') else '❌ Offline'}"
    )
else:
    st.sidebar.markdown("**API:** ❌ Unreachable")

st.sidebar.markdown("---")
st.sidebar.caption("Risk Tiers: 🔴 ≥0.70 Block · 🟡 0.40–0.70 Warn · 🟢 <0.40 Allow")


# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Prompt Injection Defense Framework")
    st.markdown("**Core question:** *How effective are different defenses against prompt injection attacks on LLMs?*")

    runs, insights_data = get_overview_data()

    if not runs:
        st.info("No simulation runs yet. Start with **Detect (Live)** or **Run Simulation**.")
        st.markdown("""
| Module | Role |
|--------|------|
| **Rule-based** | Regex + keyword patterns, zero latency |
| **ML Classifier** | Logistic regression on 12 prompt features |
| **LLM Guard** | Second Ollama model classifies risk 0–1 |
| **Ensemble** | Weighted combination → continuous risk score |
| **Actions** | hard_block / soft_warn / rewrite / safe_replace |
| **Benchmark** | Defense-ON vs defense-OFF delta metrics |
        """)
    else:
        df = pd.DataFrame(runs)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Runs", len(df))
        c2.metric("Avg ASR", f"{df['attack_success_rate'].mean():.1%}")
        c3.metric("Avg Detection", f"{df['defense_detection_rate'].mean():.1%}")
        if "false_negative_rate" in df.columns:
            c4.metric("Avg FNR", f"{df['false_negative_rate'].mean():.1%}")
        c5.metric("Avg F1", f"{df['f1_score'].mean():.3f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                df.sort_values("timestamp").tail(20),
                x="timestamp",
                y=["attack_success_rate", "defense_detection_rate", "f1_score"],
                title="Key Metrics Over Runs",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if insights_data.get("by_attack_type"):
                type_df = pd.DataFrame([
                    {"type": k, "asr": v["asr"]}
                    for k, v in insights_data["by_attack_type"].items()
                ])
                fig = px.bar(type_df, x="type", y="asr",
                             title="ASR by Attack Type (all-time)",
                             color="asr", color_continuous_scale="RdYlGn_r")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

        if insights_data.get("tier_distribution"):
            td = insights_data["tier_distribution"]
            fig = px.pie(values=list(td.values()), names=list(td.keys()),
                         title="Risk Tier Distribution (all-time)",
                         color=list(td.keys()),
                         color_discrete_map={"block": "#dc3545", "warn": "#ffc107", "allow": "#28a745"})
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Detect (Live)":
    st.title("Live Detection — Unified detect() Interface")
    st.markdown("Enter any prompt to get the full ensemble risk assessment instantly.")

    col_p, col_m = st.columns([3, 1])
    with col_p:
        prompt = st.text_area("Prompt to evaluate:", height=140,
                              placeholder="Type any prompt here…")
    with col_m:
        detect_mode = st.selectbox(
            "Defense Mode",
            ["no_llm", "rule_only", "full"],
            help="no_llm = fast (rule+ML). full = adds LLM guard (~30s).",
        )
        st.caption("🟢 rule_only — instant\n🟡 no_llm — ~1s\n🔴 full — ~30s")
    show_breakdown = st.checkbox("Show ensemble component breakdown", value=True)

    if st.button("Detect", type="primary", disabled=not prompt):
        with st.spinner("Running ensemble defense pipeline…"):
            result = api_post("/defense/detect", {"prompt": prompt, "defense_mode": detect_mode})
        if result:
            score = result["risk_score"]
            col1, col2, col3 = st.columns(3)
            col1.plotly_chart(score_gauge(score), use_container_width=True)
            col2.metric("Decision", tier_badge(result["decision"]))
            col2.metric("Label", "🚨 Malicious" if result["label"] == "malicious" else "✅ Safe")
            col3.metric("Action Taken", result.get("action_taken", "none").upper().replace("_", " "))
            col3.markdown(f"**Reason:** {result.get('reason', '—')[:80]}")

            if show_breakdown and result.get("ensemble_scores"):
                scores = {k: v for k, v in result["ensemble_scores"].items() if k != "ensemble"}
                fig = go.Figure(go.Bar(
                    x=list(scores.keys()), y=list(scores.values()),
                    marker_color=["#dc3545" if v >= 0.7 else "#ffc107" if v >= 0.4 else "#28a745"
                                  for v in scores.values()],
                ))
                fig.add_hline(y=0.70, line_dash="dash", line_color="red",
                              annotation_text="Block (0.70)")
                fig.add_hline(y=0.40, line_dash="dash", line_color="orange",
                              annotation_text="Warn (0.40)")
                fig.update_layout(title="Per-Component Scores",
                                  yaxis_range=[0, 1], height=280,
                                  margin=dict(t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            if result.get("sanitized_prompt") and result["sanitized_prompt"] != prompt:
                with st.expander("Sanitized Prompt"):
                    st.code(result["sanitized_prompt"])


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Run Simulation":
    st.title("Run Full Simulation")

    models = get_models()
    attacks_data = get_attacks()
    attack_types = list({a["type"] for a in attacks_data})

    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Victim Model", models)
        system_mode = st.selectbox("System Prompt Mode",
                                   ["standard", "hardened", "none"])
    with col2:
        selected_types = st.multiselect("Attack Types (empty = all)", attack_types)
        defense_mode = st.selectbox("Defense Mode",
                                    ["full", "no_llm", "rule_only"])
    with col3:
        st.markdown("**Defense Mode**")
        st.caption("**full** — rule + ML + LLM guard")
        st.caption("**no_llm** — rule + ML only (faster)")
        st.caption("**rule_only** — regex + keywords only")

    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Running {len(attacks_data)} attacks against {model}…"):
            result = api_post("/simulate", {
                "model": model,
                "system_prompt_mode": system_mode,
                "attack_types": selected_types or None,
                "defense_mode": defense_mode,
            })
        if result:
            st.success(f"Run ID: `{result['run_id']}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ASR", f"{result['attack_success_rate']:.1%}")
            c2.metric("Detection Rate", f"{result['defense_detection_rate']:.1%}")
            c3.metric("FNR", f"{result.get('false_negative_rate', 0):.1%}")
            c4.metric("FPR", f"{result.get('false_positive_rate', 0):.1%}")
            c5.metric("F1", f"{result['f1_score']:.3f}")

            col1, col2 = st.columns(2)
            with col1:
                cm = pd.DataFrame({
                    "Category": ["TP", "FP", "TN", "FN"],
                    "Count": [result["true_positives"], result["false_positives"],
                              result["true_negatives"], result["false_negatives"]],
                })
                fig = px.bar(cm, x="Category", y="Count", color="Category",
                             title="Defense Confusion Matrix",
                             color_discrete_map={
                                 "TP": "#28a745", "FP": "#ffc107",
                                 "TN": "#17a2b8", "FN": "#dc3545"})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if result.get("tier_distribution"):
                    td = result["tier_distribution"]
                    fig = px.pie(values=list(td.values()), names=list(td.keys()),
                                 title="Risk Tier Distribution",
                                 color=list(td.keys()),
                                 color_discrete_map={"block": "#dc3545",
                                                     "warn": "#ffc107",
                                                     "allow": "#28a745"})
                    st.plotly_chart(fig, use_container_width=True)

            if result.get("by_attack_type"):
                st.subheader("Per Attack Type")
                st.dataframe(
                    pd.DataFrame([{"type": k, **v}
                                  for k, v in result["by_attack_type"].items()]),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Single Attack":
    st.title("Test a Single Attack")

    attacks_data = get_attacks()
    models = get_models()
    mode = st.radio("Input Mode", ["From Library", "Custom Prompt"], horizontal=True)

    if mode == "From Library":
        selected = st.selectbox("Select Attack", [a["name"] for a in attacks_data])
        attack_info = next((a for a in attacks_data if a["name"] == selected), {})
        st.info(f"**Type:** {attack_info.get('type')}  |  **Severity:** {attack_info.get('severity')}")
        prompt = st.text_area("Prompt (editable)",
                              value=attack_info.get("prompt", ""), height=160)
    else:
        selected = "custom"
        attack_info = {"type": "instruction_override"}
        prompt = st.text_area("Custom Prompt", height=160)

    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", models)
        system_mode = st.selectbox("System Prompt Mode",
                                   ["standard", "hardened", "none"])

    if st.button("Execute Attack", type="primary", disabled=not prompt):
        with st.spinner("Running…"):
            result = api_post("/simulate/single", {
                "prompt": prompt,
                "model": model,
                "system_prompt_mode": system_mode,
                "attack_name": selected,
                "attack_type": attack_info.get("type", "instruction_override"),
            })
        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Attack Succeeded",
                      "YES ⚠️" if result["attack_succeeded"] else "NO ✅")
            c2.metric("Risk Tier", tier_badge(result.get("risk_tier", "allow")))
            c3.plotly_chart(score_gauge(result["risk_score"]),
                            use_container_width=True)
            c4.metric("Action",
                      result.get("action_taken", "none").upper().replace("_", " "))

            if result.get("ensemble_scores"):
                scores = {k: v for k, v in result["ensemble_scores"].items()
                          if k != "ensemble"}
                fig = go.Figure(go.Bar(
                    x=list(scores.keys()), y=list(scores.values()),
                    marker_color=["#dc3545" if v >= 0.7
                                  else "#ffc107" if v >= 0.4
                                  else "#28a745"
                                  for v in scores.values()],
                ))
                fig.add_hline(y=0.70, line_dash="dash", line_color="red")
                fig.add_hline(y=0.40, line_dash="dash", line_color="orange")
                fig.update_layout(title="Component Scores",
                                  yaxis_range=[0, 1], height=260)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("LLM Response")
            st.code(result["response"], language="text")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Adversarial Loop":
    st.title("Adversarial Attack Loop")
    st.markdown(
        "Uses a local LLM to auto-mutate prompts until one bypasses defense "
        "or iterations exhaust."
    )

    attacks_data = get_attacks()
    models = get_models()
    col1, col2, col3 = st.columns(3)
    with col1:
        seed = st.selectbox("Seed Attack", [a["name"] for a in attacks_data])
    with col2:
        model = st.selectbox("Model", models)
    with col3:
        max_iter = st.slider("Max Iterations", 1, 10, 5)

    if st.button("Start Loop", type="primary"):
        with st.spinner("Mutating and testing…"):
            result = api_post("/simulate/adversarial-loop", {
                "attack_name": seed, "model": model,
                "system_prompt_mode": "standard",
                "max_iterations": max_iter,
            })
        if result:
            iters = result.get("iterations", [])
            bypassed = [i for i in iters if i["succeeded"]]
            c1, c2 = st.columns(2)
            c1.metric("Iterations Run", len(iters))
            c2.metric("Bypassed Defense", len(bypassed))
            for it in iters:
                icon = "⚠️ BYPASSED" if it["succeeded"] else "✅ BLOCKED"
                with st.expander(
                    f"Iter {it['iteration']} — {icon}  ({it['attack_name']})"
                ):
                    st.markdown("**Prompt:**")
                    st.code(it["prompt"])
                    st.markdown("**Response:**")
                    st.code(it["response"])


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Benchmark":
    st.title("Benchmark — No Defense vs Full Defense")
    st.markdown(
        "**Answers:** *How much does defense actually reduce attack success?*  "
        "Runs each attack twice and computes delta metrics."
    )

    models = get_models()
    attacks_data = get_attacks()
    attack_types = list({a["type"] for a in attacks_data})

    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Victim Model", models)
        system_mode = st.selectbox("System Prompt Mode",
                                   ["standard", "hardened", "none"])
    with col2:
        selected_types = st.multiselect("Attack Types (empty = all)", attack_types)
        defense_mode = st.selectbox(
            "Defense Mode",
            ["no_llm", "rule_only", "full"],
            help="no_llm = rule-based + ML only (fast). full = adds LLM guard (slow).",
        )
    with col3:
        st.markdown("**Speed guide**")
        st.caption("🟢 **rule_only** — ~5 sec")
        st.caption("🟡 **no_llm** — ~15 sec (rule + ML)")
        st.caption("🔴 **full** — several minutes (adds LLM guard per prompt)")
        n_attacks = len([a for a in attacks_data
                         if not selected_types or a["type"] in selected_types])
        st.caption(f"**{n_attacks} attacks × 2 runs**")

    if st.button("Run Benchmark", type="primary"):
        est = {"rule_only": "~5 sec", "no_llm": "~15 sec", "full": "several minutes"}
        with st.spinner(
            f"Running benchmark in **{defense_mode}** mode "
            f"({est.get(defense_mode, '')})…"
        ):
            result = api_post("/benchmark", {
                "model": model,
                "system_prompt_mode": system_mode,
                "attack_types": selected_types or None,
                "defense_mode": defense_mode,
            })

        if result:
            baseline = result.get("baseline", {})
            defended = result.get("defended", {})
            delta = result.get("delta", {})
            insights = result.get("insights", [])

            # Headline delta metrics
            st.subheader("Defense Impact")
            asr_red = delta.get("asr_reduction", 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ASR Reduction", f"{asr_red:.1%}",
                      delta=f"{asr_red:+.1%}",
                      delta_color="normal" if asr_red > 0 else "inverse")
            c2.metric("Detection Rate",
                      f"{defended.get('defense_detection_rate', 0):.1%}")
            c3.metric("False Negative Rate",
                      f"{defended.get('false_negative_rate', 0):.1%}",
                      help="% of attacks that slipped through")
            c4.metric("F1 Score", f"{defended.get('f1_score', 0):.3f}")

            # Side-by-side bar
            metrics = ["attack_success_rate", "defense_detection_rate",
                       "false_negative_rate", "false_positive_rate", "f1_score"]
            labels = ["ASR", "Detection Rate", "FNR", "FPR", "F1"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="No Defense (Baseline)", x=labels,
                y=[baseline.get(m, 0) for m in metrics],
                marker_color="#dc3545"))
            fig.add_trace(go.Bar(
                name="With Defense", x=labels,
                y=[defended.get(m, 0) for m in metrics],
                marker_color="#28a745"))
            fig.update_layout(barmode="group", yaxis_range=[0, 1],
                              title="Defense Effectiveness Comparison")
            st.plotly_chart(fig, use_container_width=True)

            # Per-type comparison
            col1, col2 = st.columns(2)
            with col1:
                if baseline.get("by_attack_type"):
                    base_rows = [{"type": k, "baseline_asr": v["asr"]}
                                 for k, v in baseline["by_attack_type"].items()]
                    def_rows = [{"type": k, "defended_asr": v["asr"]}
                                for k, v in defended.get("by_attack_type", {}).items()]
                    merged = (pd.DataFrame(base_rows)
                              .merge(pd.DataFrame(def_rows), on="type", how="outer")
                              .fillna(0))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Baseline ASR", x=merged["type"],
                                        y=merged["baseline_asr"],
                                        marker_color="#dc3545"))
                    fig.add_trace(go.Bar(name="Defended ASR", x=merged["type"],
                                        y=merged["defended_asr"],
                                        marker_color="#28a745"))
                    fig.update_layout(barmode="group", title="ASR by Attack Type",
                                      yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if defended.get("tier_distribution"):
                    td = defended["tier_distribution"]
                    fig = px.pie(values=list(td.values()), names=list(td.keys()),
                                 title="Risk Tier Distribution (Defended)",
                                 color=list(td.keys()),
                                 color_discrete_map={"block": "#dc3545",
                                                     "warn": "#ffc107",
                                                     "allow": "#28a745"})
                    st.plotly_chart(fig, use_container_width=True)

            # Auto-generated insights
            if insights:
                st.subheader("Auto-generated Security Insights")
                for ins in insights:
                    icon = "⚠️" if any(w in ins for w in
                                       ["HIGH", "LOW", "limited", "room"]) else "✅"
                    st.markdown(f"{icon} {ins}")

            with st.expander("Full Metric Tables"):
                comp_df = pd.DataFrame({
                    "Metric": labels,
                    "No Defense": [baseline.get(m, 0) for m in metrics],
                    "With Defense": [defended.get(m, 0) for m in metrics],
                })
                st.dataframe(comp_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Compare Models":
    st.title("Multi-Model Vulnerability Comparison")
    st.markdown("Which model is more resistant to prompt injection?")

    models = get_models()
    attacks_data = get_attacks()
    attack_types = list({a["type"] for a in attacks_data})
    selected_models = st.multiselect("Models to Compare", models,
                                     default=models[:2] if len(models) >= 2 else models)
    selected_types = st.multiselect("Attack Types (empty = all)", attack_types)

    if st.button("Compare", type="primary", disabled=len(selected_models) < 2):
        with st.spinner("Running against each model…"):
            result = api_post("/simulate/compare-models", {
                "models": selected_models,
                "system_prompt_mode": "standard",
                "attack_types": selected_types or None,
            })
        if result:
            comparison = result.get("comparison", {})
            rows = [{"model": m, **s} for m, s in comparison.items()]
            df = pd.DataFrame(rows)
            key_cols = ["model", "attack_success_rate", "defense_detection_rate",
                        "false_negative_rate", "f1_score"]
            st.dataframe(df[[c for c in key_cols if c in df.columns]],
                         use_container_width=True)

            plot_metrics = ["attack_success_rate", "defense_detection_rate",
                            "false_negative_rate", "f1_score"]
            plot_labels = ["ASR", "Detection Rate", "FNR", "F1"]
            fig = go.Figure()
            for _, row in df.iterrows():
                fig.add_trace(go.Bar(
                    name=row["model"], x=plot_labels,
                    y=[row.get(m, 0) for m in plot_metrics],
                ))
            fig.update_layout(barmode="group", yaxis_range=[0, 1],
                              title="Model Vulnerability Comparison")
            st.plotly_chart(fig, use_container_width=True)

            most_vulnerable = df.loc[df["attack_success_rate"].idxmax(), "model"]
            st.warning(f"Most vulnerable model: **{most_vulnerable}** (highest ASR)")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Insights":
    st.title("Security Insights — All-Time Analytics")

    if st.button("🔄 Refresh", key="refresh_insights"):
        get_insights_data.clear()
    data = get_insights_data()
    if not data or not data.get("total_attacks"):
        st.info("No data yet. Run some simulations first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Attacks Logged", data["total_attacks"])
        c2.metric("Overall ASR", f"{data.get('overall_asr', 0):.1%}")
        c3.metric("Overall Block Rate", f"{data.get('overall_block_rate', 0):.1%}")

        col1, col2 = st.columns(2)
        with col1:
            if data.get("by_attack_type"):
                df = pd.DataFrame([
                    {"type": k, "asr": v["asr"], "avg_risk": v["avg_risk"]}
                    for k, v in data["by_attack_type"].items()
                ])
                fig = px.bar(df, x="type", y="asr", color="asr",
                             color_continuous_scale="RdYlGn_r",
                             title="Attack Success Rate by Type")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if data.get("tier_distribution"):
                td = data["tier_distribution"]
                fig = px.pie(values=list(td.values()), names=list(td.keys()),
                             title="All-Time Risk Tier Distribution",
                             color=list(td.keys()),
                             color_discrete_map={"block": "#dc3545",
                                                 "warn": "#ffc107",
                                                 "allow": "#28a745"})
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if data.get("by_model"):
                model_df = pd.DataFrame([
                    {"model": k, "asr": v["asr"]}
                    for k, v in data["by_model"].items()
                ])
                fig = px.bar(model_df, x="model", y="asr",
                             title="ASR by Model (vulnerability ranking)",
                             color="asr", color_continuous_scale="RdYlGn_r")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if data.get("top_bypassing_attacks"):
                bypass_df = pd.DataFrame(data["top_bypassing_attacks"])
                fig = px.bar(bypass_df, x="attack", y="count",
                             title="Top Attacks That Bypassed Defense",
                             color="count", color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Logs":
    st.title("Simulation Logs")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        run_id_filter = st.text_input("Run ID")
    with col2:
        model_filter = st.selectbox("Model", [""] + get_models())
    with col3:
        type_filter = st.selectbox("Attack Type",
                                   ["", "instruction_override", "jailbreak",
                                    "data_exfiltration", "indirect_injection"])
    with col4:
        tier_filter = st.selectbox("Risk Tier", ["", "block", "warn", "allow"])
    with col5:
        limit = st.number_input("Limit", 10, 1000, 200)

    if st.button("Load Logs"):
        params = {"limit": limit}
        if run_id_filter:
            params["run_id"] = run_id_filter
        if model_filter:
            params["model"] = model_filter
        if type_filter:
            params["attack_type"] = type_filter
        if tier_filter:
            params["risk_tier"] = tier_filter

        logs = api_get("/logs", params=params) or []
        if logs:
            df = pd.DataFrame(logs)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x="defense_risk_score", nbins=20,
                                   color="risk_tier" if "risk_tier" in df.columns else None,
                                   color_discrete_map={"block": "#dc3545",
                                                       "warn": "#ffc107",
                                                       "allow": "#28a745"},
                                   title="Risk Score Distribution")
                fig.add_vline(x=0.70, line_dash="dash", line_color="red",
                              annotation_text="Block")
                fig.add_vline(x=0.40, line_dash="dash", line_color="orange",
                              annotation_text="Warn")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                asr_df = df.groupby("attack_type")["attack_succeeded"].mean().reset_index()
                asr_df.columns = ["attack_type", "asr"]
                fig = px.bar(asr_df, x="attack_type", y="asr",
                             title="ASR by Attack Type", color="asr",
                             color_continuous_scale="RdYlGn_r")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            if "defense_action" in df.columns:
                action_counts = df["defense_action"].value_counts().reset_index()
                action_counts.columns = ["action", "count"]
                fig = px.pie(action_counts, names="action", values="count",
                             title="Defense Actions Taken")
                st.plotly_chart(fig, use_container_width=True)

            display_cols = ["id", "timestamp", "model", "attack_name", "attack_type",
                            "attack_succeeded", "defense_blocked", "risk_tier",
                            "defense_action", "defense_risk_score", "latency_ms"]
            st.dataframe(df[[c for c in display_cols if c in df.columns]],
                         use_container_width=True)

            if st.checkbox("Show prompts & responses"):
                for _, row in df.iterrows():
                    tier = row.get("risk_tier", "?")
                    with st.expander(
                        f"[{tier.upper()}] {row['attack_name']} · {row['timestamp']}"
                    ):
                        rs = row["defense_risk_score"]
                        st.markdown(f"**Risk Score:** {risk_color(rs)} `{rs:.3f}`")
                        st.markdown("**Prompt:**")
                        st.code(row["prompt"])
                        st.markdown("**Response:**")
                        st.code(row["response"])
        else:
            st.info("No logs matched the filters.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Replay":
    st.title("Replay a Logged Attack")
    st.markdown(
        "Re-run a previously logged attack to check if the defense outcome changes "
        "after tuning thresholds or enabling/disabling components."
    )

    if "replay_logs" not in st.session_state:
        st.session_state.replay_logs = []

    col_load, _ = st.columns([1, 3])
    with col_load:
        if st.button("Load Recent Logs"):
            st.session_state.replay_logs = api_get("/logs", params={"limit": 100}) or []

    logs = st.session_state.replay_logs
    if not logs:
        st.info("Click **Load Recent Logs** to fetch logged attacks for replay.")
    else:
        df = pd.DataFrame(logs)
        df["label"] = df.apply(
            lambda r: (
                f"[{r.get('id')}] {r['attack_name']} | "
                f"tier={r.get('risk_tier','?')} | "
                f"succeeded={bool(r['attack_succeeded'])}"
            ),
            axis=1,
        )
        selected_label = st.selectbox("Select a logged attack to replay",
                                      df["label"].tolist())
        log_id = int(selected_label.split("]")[0].strip("["))

        col1, col2 = st.columns(2)
        with col1:
            replay_model = st.selectbox("Override Model",
                                        ["(original)"] + get_models())
        with col2:
            defense_mode = st.selectbox("Defense Mode",
                                        ["full", "no_llm", "rule_only"])

        if st.button("Replay Attack", type="primary"):
            with st.spinner("Replaying…"):
                payload = {"log_id": log_id, "defense_mode": defense_mode}
                if replay_model != "(original)":
                    payload["model"] = replay_model
                result = api_post("/replay", payload)

            if result:
                orig = result.get("original_result", {})
                replay = result.get("replay_result", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Run")
                    st.metric("Succeeded",
                              "YES ⚠️" if orig.get("attack_succeeded") else "NO ✅")
                    st.metric("Blocked",
                              "YES ✅" if orig.get("defense_blocked") else "NO ⚠️")
                    st.metric("Risk Score", f"{orig.get('risk_score', 0):.3f}")
                with col2:
                    st.subheader("Replay Result")
                    st.metric("Succeeded",
                              "YES ⚠️" if replay.get("attack_succeeded") else "NO ✅")
                    st.metric("Blocked",
                              "YES ✅" if replay.get("defense_blocked") else "NO ⚠️")
                    st.metric("Risk Score", f"{replay.get('risk_score', 0):.3f}")
                    st.metric("Risk Tier", tier_badge(replay.get("risk_tier", "allow")))
                    st.metric("Action", replay.get("action_taken", "none").upper())
                st.subheader("Replay LLM Response")
                st.code(replay.get("response", ""), language="text")
