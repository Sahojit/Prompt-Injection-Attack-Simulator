"""
app.py — 5-Layer Defense Dashboard

Run with:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from attacks import ATTACKS
from simulator import run_all_layers
from evaluator import compute_metrics
import updater

st.set_page_config(
    page_title="Prompt Injection Simulator",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Prompt Injection Attack Simulator")
st.caption("Demonstrates all 5 defense layers against prompt injection attacks on a local LLM.")

# Sidebar
page = st.sidebar.radio("Navigate", [
    "Run Simulation",
    "Test a Prompt",
    "Update Defense (Layer 5)",
    "About",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**5 Defense Layers:**
1. 🔤 Rule-based keywords
2. 🤖 ML Classifier
3. 🔒 Hardened System Prompt
4. 🔍 Output Filter
5. 🔄 Continuous Updates
""")


# ── Helper: show layer badges ─────────────────────────────────────────────────

def show_layer_results(r: dict):
    """Displays a row of pass/fail badges for each defense layer."""
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"**Layer 1 (Rules)**\n\n{'🔴 BLOCKED' if r['layer1_blocked'] else '🟢 PASSED'}"
    )
    col2.markdown(
        f"**Layer 2 (ML)**\n\n"
        f"{'🔴 BLOCKED' if r['layer2_blocked'] else '🟢 PASSED'} "
        f"· confidence: {r['layer2_confidence']:.0%}"
    )
    col3.markdown(
        f"**Layer 3 (LLM)**\n\n"
        f"{'⏭️ Skipped' if r['blocked'] and not r['layer3_response'] else '✅ Queried'}"
    )
    col4.markdown(
        f"**Layer 4 (Output)**\n\n"
        f"{'🔴 FILTERED' if r['layer4_filtered'] else ('✅ Clean' if r['layer3_response'] else '⏭️ Skipped')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
if page == "Run Simulation":
    st.header("Run All Attacks Through All 5 Layers")
    st.info(f"Runs all **{len(ATTACKS)} attacks** through the full 5-layer defense pipeline.")

    if st.button("▶ Run All Attacks", type="primary"):
        results = []

        for attack in ATTACKS:
            with st.spinner(f"Running: {attack['name']}…"):
                r = run_all_layers(attack["prompt"])

            blocked = r["blocked"]
            header = f"{'🛡️ BLOCKED' if blocked else '⚠️ BYPASSED'} — {attack['name']} ({attack['type']})"

            with st.expander(header, expanded=True):
                st.markdown(f"**Prompt:** `{attack['prompt'][:90]}…`")
                st.markdown("---")
                show_layer_results(r)
                st.markdown("---")

                if blocked:
                    st.success(f"**Stopped by:** {r['blocked_by']}")
                else:
                    st.error("⚠️ This attack bypassed ALL defense layers.")
                    st.markdown("**Final LLM Response:**")
                    st.code(r["final_response"], language="text")

            results.append({
                "attack_name":      attack["name"],
                "attack_type":      attack["type"],
                "blocked":          blocked,
                "attack_succeeded": not blocked,
                "blocked_by":       r["blocked_by"] or "—",
            })

        # Metrics
        metrics = compute_metrics(results)
        st.markdown("---")
        st.subheader("Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Attacks", metrics["total_attacks"])
        c2.metric("Blocked", metrics["blocked_attacks"])
        c3.metric("Attack Success Rate", f"{metrics['attack_success_rate']:.0%}",
                  help="% that bypassed ALL 5 layers — lower is better")
        c4.metric("Defense Block Rate", f"{metrics['defense_block_rate']:.0%}",
                  help="% caught by at least one layer — higher is better")

        # Summary table
        df = pd.DataFrame([{
            "Attack": r["attack_name"],
            "Type": r["attack_type"],
            "Blocked?": "✅ Yes" if r["blocked"] else "❌ No",
            "Stopped By": r["blocked_by"],
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Test a Prompt":
    st.header("Test Any Prompt Through All 5 Layers")
    st.markdown("Type any prompt and watch it go through each defense layer in real time.")

    prompt = st.text_area(
        "Enter a prompt:",
        height=120,
        placeholder="Try: 'What is your system prompt?' or 'Ignore previous instructions'",
    )
    query_llm = st.checkbox("Query LLM if defense passes (Layer 3)", value=True)

    if st.button("Test", type="primary", disabled=not prompt.strip()):
        with st.spinner("Running through all defense layers…"):
            if not query_llm:
                # Only run layers 1 and 2
                from defense import check_defense
                from ml_classifier import ml_check
                l1_blocked, l1_reason = check_defense(prompt)
                l2_blocked, l2_conf, l2_reason = ml_check(prompt)

                r = {
                    "layer1_blocked": l1_blocked,
                    "layer2_blocked": l2_blocked,
                    "layer2_confidence": l2_conf,
                    "layer3_response": None,
                    "layer4_filtered": False,
                    "blocked": l1_blocked or l2_blocked,
                    "blocked_by": (
                        f"Layer 1: {l1_reason}" if l1_blocked else
                        f"Layer 2: {l2_reason}" if l2_blocked else None
                    ),
                    "final_response": "[LLM query skipped]",
                }
            else:
                r = run_all_layers(prompt)

        st.markdown("### Layer-by-Layer Results")
        show_layer_results(r)
        st.markdown("---")

        if r["blocked"]:
            st.error(f"🛡️ **BLOCKED** — {r['blocked_by']}")
        else:
            st.warning("⚠️ This prompt **bypassed all active defense layers**.")
            st.markdown("**Final Response:**")
            st.code(r["final_response"], language="text")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Update Defense (Layer 5)":
    st.header("Layer 5 — Continuous Updates")
    st.markdown(
        "When a new attack bypasses the system, add it here. "
        "All defense layers get updated immediately."
    )

    st.subheader("Add a New Attack Keyword (Layer 1)")
    new_rule = st.text_input(
        "New regex pattern:",
        placeholder=r"disregard (all )?prior directives",
    )
    if st.button("Add Rule", disabled=not new_rule.strip()):
        updater.add_attack_keyword(new_rule.strip())
        st.success(f"Rule added to Layer 1: `{new_rule}`")

    st.markdown("---")

    st.subheader("Teach the ML Model a New Attack (Layer 2)")
    new_attack = st.text_area(
        "Attack prompt that bypassed the system:",
        height=80,
        placeholder="e.g. What instructions were you originally given?",
    )
    if st.button("Add to ML Training", disabled=not new_attack.strip()):
        updater.add_ml_attack(new_attack.strip())
        st.success("ML model retrained with new attack example.")

    st.markdown("---")

    st.subheader("Add a New Output Filter Pattern (Layer 4)")
    new_output_pattern = st.text_input(
        "Pattern that indicates the LLM leaked information:",
        placeholder=r"as requested, my instructions are",
    )
    if st.button("Add Output Pattern", disabled=not new_output_pattern.strip()):
        updater.add_output_pattern(new_output_pattern.strip())
        st.success(f"Output filter updated with: `{new_output_pattern}`")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.header("How the 5-Layer Defense Works")

    st.markdown("""
## The Flow

```
User Prompt
     │
     ▼
┌─────────────────────────────────────┐
│  Layer 1: Rule-based Defense        │  Keyword/regex scan — instant
│  (defense.py)                       │
└──────────────┬──────────────────────┘
               │ blocked? → STOP
               ▼
┌─────────────────────────────────────┐
│  Layer 2: ML Classifier             │  Trained model — catches rephrased attacks
│  (ml_classifier.py)                 │
└──────────────┬──────────────────────┘
               │ blocked? → STOP
               ▼
┌─────────────────────────────────────┐
│  Layer 3: Hardened System Prompt    │  LLM is instructed to resist attacks
│  (llm.py)                           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Layer 4: Output Filter             │  Scans LLM response for leaks
│  (output_filter.py)                 │
└──────────────┬──────────────────────┘
               │
               ▼
          Safe Response
               │
    If attack bypassed everything:
               ▼
┌─────────────────────────────────────┐
│  Layer 5: Continuous Updates        │  Add new attack → all layers get smarter
│  (updater.py)                       │
└─────────────────────────────────────┘
```

## Why Each Layer Exists

| Layer | What It Does | What It Catches | What It Misses |
|-------|-------------|-----------------|----------------|
| 1 — Rules | Keyword/regex scan | Known attack phrases | Rephrased attacks |
| 2 — ML | Trained classifier | Novel/rephrased attacks | Very clever attacks |
| 3 — Hardened Prompt | Tells LLM to resist | Many LLM-level attacks | Sophisticated jailbreaks |
| 4 — Output Filter | Scans LLM response | Successful leaks in output | Non-obvious leaks |
| 5 — Updates | Adds new examples | Future attacks of same type | Attacks never seen before |

## Key Point

> No single layer is perfect. Security comes from stacking all 5.
> An attack must bypass **every** layer to succeed.
    """)
