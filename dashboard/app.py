"""
Streamlit Dashboard for the Prompt Injection Attack Simulator.
Connects to the FastAPI backend at localhost:8000.
"""
import time
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Prompt Injection Simulator",
    page_icon="🔐",
    layout="wide",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def risk_badge(score: float) -> str:
    if score >= 0.9:
        return f"🔴 {score:.2f}"
    if score >= 0.6:
        return f"🟠 {score:.2f}"
    if score >= 0.3:
        return f"🟡 {score:.2f}"
    return f"🟢 {score:.2f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔐 Attack Simulator")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Run Simulation", "Single Attack", "Adversarial Loop",
     "Compare Models", "Defense Tester", "Logs"],
)

health = api_get("/health")
if health:
    ollama_ok = health.get("ollama_available", False)
    st.sidebar.markdown(f"**Ollama:** {'✅ Online' if ollama_ok else '❌ Offline'}")
else:
    st.sidebar.markdown("**API:** ❌ Unreachable")

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("Prompt Injection Attack Simulator")
    st.markdown("**Local LLM Security Testing Framework** — powered by Ollama")

    runs = api_get("/logs/runs") or []
    if not runs:
        st.info("No simulation runs yet. Go to 'Run Simulation' to get started.")
    else:
        df = pd.DataFrame(runs)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Runs", len(df))
        col2.metric("Avg Attack Success Rate", f"{df['attack_success_rate'].mean():.1%}")
        col3.metric("Avg Defense Detection", f"{df['defense_detection_rate'].mean():.1%}")
        col4.metric("Avg F1 Score", f"{df['f1_score'].mean():.3f}")

        st.subheader("Run History")
        st.dataframe(
            df[["run_id", "timestamp", "total_attacks", "attack_success_rate",
                "defense_detection_rate", "f1_score"]].head(20),
            use_container_width=True,
        )

        fig = px.line(
            df.sort_values("timestamp").tail(20),
            x="timestamp", y=["attack_success_rate", "defense_detection_rate", "f1_score"],
            title="Metrics Over Runs",
        )
        st.plotly_chart(fig, use_container_width=True)


elif page == "Run Simulation":
    st.title("Run Full Simulation")

    models_data = api_get("/models") or {}
    available_models = models_data.get("models", ["llama3", "mistral"])
    attacks_data = api_get("/attacks") or []
    attack_types = list({a["type"] for a in attacks_data})

    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Victim Model", available_models)
        system_mode = st.selectbox("System Prompt Mode", ["standard", "hardened", "none"])
    with col2:
        selected_types = st.multiselect("Attack Types (empty = all)", attack_types)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation against Ollama..."):
            result = api_post("/simulate", {
                "model": model,
                "system_prompt_mode": system_mode,
                "attack_types": selected_types or None,
            })
        if result:
            st.success(f"Run ID: `{result['run_id']}`")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Attack Success Rate", f"{result['attack_success_rate']:.1%}")
            col2.metric("Defense Detection", f"{result['defense_detection_rate']:.1%}")
            col3.metric("F1 Score", f"{result['f1_score']:.3f}")
            col4.metric("Total Attacks", result["total_attacks"])

            # confusion matrix
            cm_data = {
                "Metric": ["True Positives", "False Positives", "True Negatives", "False Negatives"],
                "Count": [result["true_positives"], result["false_positives"],
                          result["true_negatives"], result["false_negatives"]],
            }
            fig = px.bar(pd.DataFrame(cm_data), x="Metric", y="Count",
                         color="Metric", title="Defense Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)


elif page == "Single Attack":
    st.title("Test a Single Attack")

    attacks_data = api_get("/attacks") or []
    attack_names = [a["name"] for a in attacks_data]
    attack_types_list = [t.value for t in __import__("src.attacks.attack_types", fromlist=["AttackType"]).AttackType]

    mode = st.radio("Input Mode", ["From Library", "Custom Prompt"])

    if mode == "From Library":
        selected = st.selectbox("Select Attack", attack_names)
        attack_info = next((a for a in attacks_data if a["name"] == selected), {})
        st.info(f"**Type:** {attack_info.get('type')} | **Severity:** {attack_info.get('severity')}")
        prompt = st.text_area("Prompt", value=attack_info.get("prompt_preview", ""), height=150)
    else:
        prompt = st.text_area("Custom Prompt", height=150)
        selected = "custom"

    col1, col2 = st.columns(2)
    with col1:
        models_data = api_get("/models") or {}
        model = st.selectbox("Model", models_data.get("models", ["llama3"]))
    with col2:
        system_mode = st.selectbox("System Prompt Mode", ["standard", "hardened", "none"])

    if st.button("Execute Attack", type="primary") and prompt:
        with st.spinner("Attacking..."):
            result = api_post("/simulate/single", {
                "prompt": prompt,
                "model": model,
                "system_prompt_mode": system_mode,
                "attack_name": selected,
                "attack_type": attack_info.get("type", "instruction_override") if mode == "From Library" else "instruction_override",
            })
        if result:
            col1, col2, col3 = st.columns(3)
            col1.metric("Attack Succeeded", "YES ⚠️" if result["attack_succeeded"] else "NO ✅")
            col2.metric("Defense Blocked", "YES ✅" if result["defense_blocked"] else "NO ⚠️")
            col3.metric("Risk Score", risk_badge(result["risk_score"]))
            st.subheader("LLM Response")
            st.code(result["response"], language="text")


elif page == "Adversarial Loop":
    st.title("Adversarial Attack Loop")
    st.markdown("Auto-generates mutations of a seed attack until one succeeds or iterations exhaust.")

    attacks_data = api_get("/attacks") or []
    attack_names = [a["name"] for a in attacks_data]

    col1, col2, col3 = st.columns(3)
    with col1:
        seed = st.selectbox("Seed Attack", attack_names)
    with col2:
        models_data = api_get("/models") or {}
        model = st.selectbox("Model", models_data.get("models", ["llama3"]))
    with col3:
        max_iter = st.slider("Max Iterations", 1, 10, 5)

    if st.button("Start Loop", type="primary"):
        with st.spinner("Running adversarial loop..."):
            result = api_post("/simulate/adversarial-loop", {
                "attack_name": seed,
                "model": model,
                "system_prompt_mode": "standard",
                "max_iterations": max_iter,
            })
        if result:
            iterations = result.get("iterations", [])
            st.success(f"Completed {len(iterations)} iterations")
            for it in iterations:
                color = "red" if it["succeeded"] else "green"
                icon = "⚠️ SUCCEEDED" if it["succeeded"] else "✅ BLOCKED"
                with st.expander(f"Iteration {it['iteration']} — {icon} ({it['attack_name']})"):
                    st.markdown(f"**Prompt:**\n```\n{it['prompt']}\n```")
                    st.markdown(f"**Response:**\n```\n{it['response']}\n```")


elif page == "Compare Models":
    st.title("Multi-Model Comparison")

    models_data = api_get("/models") or {}
    available = models_data.get("models", ["llama3", "mistral"])
    attacks_data = api_get("/attacks") or []
    attack_types = list({a["type"] for a in attacks_data})

    selected_models = st.multiselect("Models to Compare", available, default=available[:2])
    selected_types = st.multiselect("Attack Types (empty = all)", attack_types)

    if st.button("Compare", type="primary") and len(selected_models) >= 2:
        with st.spinner("Running comparison..."):
            result = api_post("/simulate/compare-models", {
                "models": selected_models,
                "system_prompt_mode": "standard",
                "attack_types": selected_types or None,
            })
        if result:
            comparison = result.get("comparison", {})
            rows = []
            for model, metrics in comparison.items():
                rows.append({"model": model, **metrics})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            metrics_to_plot = ["attack_success_rate", "defense_detection_rate", "f1_score"]
            fig = go.Figure()
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(name=metric, x=df["model"], y=df[metric]))
            fig.update_layout(barmode="group", title="Model Comparison")
            st.plotly_chart(fig, use_container_width=True)


elif page == "Defense Tester":
    st.title("Defense Layer Tester")
    st.markdown("Test any prompt against all three defense strategies in real time.")

    prompt = st.text_area("Enter a prompt to evaluate:", height=150)

    if st.button("Evaluate Defenses", type="primary") and prompt:
        with st.spinner("Evaluating..."):
            result = api_post("/defense/evaluate", {"prompt": prompt})
        if result:
            col1, col2 = st.columns(2)
            col1.metric("Final Decision", "BLOCKED 🛡️" if result["final_blocked"] else "ALLOWED ✅")
            col2.metric("Risk Score", risk_badge(result["final_risk_score"]))

            if result.get("sanitized_prompt") != prompt:
                st.subheader("Sanitized Prompt")
                st.code(result["sanitized_prompt"], language="text")

            st.subheader("Per-Strategy Results")
            for strategy, data in result.get("per_strategy", {}).items():
                icon = "🚫" if data["blocked"] else "✅"
                with st.expander(f"{icon} {strategy} — Risk: {data['risk_score']:.2f}"):
                    st.markdown(f"**Reason:** {data['reason']}")


elif page == "Logs":
    st.title("Simulation Logs")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        run_id_filter = st.text_input("Run ID")
    with col2:
        models_data = api_get("/models") or {}
        model_filter = st.selectbox("Model", [""] + models_data.get("models", []))
    with col3:
        type_filter = st.selectbox("Attack Type", ["", "instruction_override",
                                                     "jailbreak", "data_exfiltration", "indirect_injection"])
    with col4:
        limit = st.number_input("Limit", 10, 1000, 100)

    if st.button("Load Logs"):
        params = {"limit": limit}
        if run_id_filter:
            params["run_id"] = run_id_filter
        if model_filter:
            params["model"] = model_filter
        if type_filter:
            params["attack_type"] = type_filter

        logs = api_get("/logs", params=params) or []
        if logs:
            df = pd.DataFrame(logs)

            # summary charts
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, names="attack_type", title="Attacks by Type")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                success_df = df.groupby("attack_type")["attack_succeeded"].mean().reset_index()
                fig = px.bar(success_df, x="attack_type", y="attack_succeeded",
                             title="Success Rate by Attack Type")
                st.plotly_chart(fig, use_container_width=True)

            # risk score distribution
            fig = px.histogram(df, x="defense_risk_score", nbins=20,
                               title="Risk Score Distribution", color="attack_succeeded")
            st.plotly_chart(fig, use_container_width=True)

            # raw table
            st.subheader("Raw Logs")
            display_cols = ["timestamp", "model", "attack_name", "attack_type",
                            "attack_succeeded", "defense_blocked", "defense_risk_score", "latency_ms"]
            st.dataframe(df[display_cols], use_container_width=True)

            if st.checkbox("Show full prompts/responses"):
                for _, row in df.iterrows():
                    with st.expander(f"{row['attack_name']} | {row['timestamp']}"):
                        st.markdown(f"**Prompt:** {row['prompt']}")
                        st.markdown(f"**Response:** {row['response']}")
        else:
            st.info("No logs found.")
