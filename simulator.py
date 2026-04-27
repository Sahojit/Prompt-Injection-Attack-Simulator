"""
simulator.py — Runs all 5 defense layers in sequence.

FULL FLOW:
  Attack Prompt
      │
      ▼
  Layer 1: Rule-based check      (defense.py)
      │ blocked? → STOP
      ▼
  Layer 2: ML classifier check   (ml_classifier.py)
      │ blocked? → STOP
      ▼
  Layer 3: Hardened system prompt → LLM call  (llm.py)
      │
      ▼
  Layer 4: Output filter         (output_filter.py)
      │ filtered? → replace with safe message
      ▼
  Final Response shown to user

  Layer 5: If attack bypassed everything → call updater to add it
           so future attacks of this type get caught.

Run directly:
  python simulator.py
"""

from attacks import ATTACKS
from defense import check_defense
from ml_classifier import ml_check
from llm import generate_response
from output_filter import filter_output
from evaluator import compute_metrics


def run_all_layers(prompt: str) -> dict:
    """
    Runs a single prompt through all 5 defense layers.
    Returns a result dict with what happened at each layer.
    """
    result = {
        "prompt": prompt,
        "layer1_blocked": False,
        "layer2_blocked": False,
        "layer2_confidence": 0.0,
        "layer3_response": None,
        "layer4_filtered": False,
        "final_response": None,
        "blocked": False,
        "blocked_by": None,
    }

    # ── Layer 1: Rule-based ───────────────────────────────────────────────────
    l1_blocked, l1_reason = check_defense(prompt)
    result["layer1_blocked"] = l1_blocked

    if l1_blocked:
        result["blocked"] = True
        result["blocked_by"] = f"Layer 1 (Rules): {l1_reason}"
        result["final_response"] = "[BLOCKED BY RULE-BASED DEFENSE]"
        return result

    # ── Layer 2: ML Classifier ────────────────────────────────────────────────
    l2_blocked, l2_confidence, l2_reason = ml_check(prompt)
    result["layer2_blocked"] = l2_blocked
    result["layer2_confidence"] = l2_confidence

    if l2_blocked:
        result["blocked"] = True
        result["blocked_by"] = f"Layer 2 (ML): {l2_reason}"
        result["final_response"] = "[BLOCKED BY ML CLASSIFIER]"
        return result

    # ── Layer 3: Hardened System Prompt + LLM ────────────────────────────────
    # Prompt passed both checks — send to LLM with hardened system prompt
    raw_response = generate_response(prompt)
    result["layer3_response"] = raw_response

    # ── Layer 4: Output Filter ────────────────────────────────────────────────
    final_response, was_filtered, filter_reason = filter_output(raw_response)
    result["layer4_filtered"] = was_filtered
    result["final_response"] = final_response

    if was_filtered:
        result["blocked"] = True
        result["blocked_by"] = f"Layer 4 (Output Filter): {filter_reason}"

    return result


def run_simulation(attacks: list[dict] | None = None) -> tuple[list[dict], dict]:
    """
    Runs all attacks through all 5 layers and computes evaluation metrics.
    """
    if attacks is None:
        attacks = ATTACKS

    results = []

    for attack in attacks:
        print(f"\n{'='*65}")
        print(f"[ATTACK]  {attack['name']}  ({attack['type']})")
        print(f"[PROMPT]  {attack['prompt'][:80]}...")

        layer_result = run_all_layers(attack["prompt"])

        if layer_result["blocked"]:
            print(f"[RESULT]  ✅ BLOCKED — {layer_result['blocked_by']}")
        else:
            print(f"[RESULT]  ⚠️  BYPASSED ALL LAYERS")
            print(f"[LLM]     {layer_result['final_response'][:120]}...")

            # Layer 5: Auto-update — new bypassing attack gets added to ML training
            from updater import update_all
            update_all(attack["prompt"])

        results.append({
            "attack_name":      attack["name"],
            "attack_type":      attack["type"],
            "prompt":           attack["prompt"],
            "blocked":          layer_result["blocked"],
            "blocked_by":       layer_result["blocked_by"],
            "layer1_blocked":   layer_result["layer1_blocked"],
            "layer2_blocked":   layer_result["layer2_blocked"],
            "layer2_confidence":layer_result["layer2_confidence"],
            "layer4_filtered":  layer_result["layer4_filtered"],
            "final_response":   layer_result["final_response"],
            "attack_succeeded": not layer_result["blocked"],
        })

    # Metrics
    metrics = compute_metrics(results)
    print(f"\n{'='*65}")
    print("FINAL RESULTS (All 5 Layers)")
    print(f"  Total attacks:       {metrics['total_attacks']}")
    print(f"  Blocked:             {metrics['blocked_attacks']}")
    print(f"  Attack success rate: {metrics['attack_success_rate']:.0%}")
    print(f"  Defense block rate:  {metrics['defense_block_rate']:.0%}")
    print(f"{'='*65}\n")

    return results, metrics


if __name__ == "__main__":
    run_simulation()
