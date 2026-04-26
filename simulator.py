"""
simulator.py — Main simulation runner.

FLOW:
  For each attack prompt:
    1. Run defense check (keyword/regex scan)
    2. If blocked → record result, skip LLM call
    3. If allowed → send to LLM, record response
  After all attacks → compute and print metrics

Run directly:
  python simulator.py
"""

from attacks import ATTACKS
from defense import check_defense
from llm import generate_response
from evaluator import compute_metrics


def run_simulation(attacks: list[dict] | None = None) -> tuple[list[dict], dict]:
    """
    Runs the full simulation for every attack in the list.

    Args:
        attacks: List of attack dicts (defaults to the full ATTACKS library).

    Returns:
        (results, metrics)
          results — one dict per attack with all details
          metrics — aggregated evaluation numbers
    """
    if attacks is None:
        attacks = ATTACKS

    results = []

    for attack in attacks:
        print(f"\n{'='*60}")
        print(f"[ATTACK]  {attack['name']}  ({attack['type']})")
        print(f"[PROMPT]  {attack['prompt'][:80]}...")

        # Step 1: defense check — no LLM involved, instant
        blocked, reason = check_defense(attack["prompt"])

        if blocked:
            # Defense caught it — LLM never sees this prompt
            response = "[BLOCKED BY DEFENSE]"
            print(f"[DEFENSE] ✅  {reason}")
        else:
            # Defense passed — send to LLM
            print("[DEFENSE] ⚠️  Passed — querying LLM...")
            response = generate_response(attack["prompt"])
            print(f"[LLM]     {response[:120]}...")

        results.append({
            "attack_name":     attack["name"],
            "attack_type":     attack["type"],
            "prompt":          attack["prompt"],
            "blocked":         blocked,
            "block_reason":    reason,
            "response":        response,
            # An attack "succeeded" if it bypassed the defense and reached the LLM
            "attack_succeeded": not blocked,
        })

    # Step 2: compute metrics
    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"  Total attacks:       {metrics['total_attacks']}")
    print(f"  Blocked by defense:  {metrics['blocked_attacks']}")
    print(f"  Attack success rate: {metrics['attack_success_rate']:.0%}  (bypassed defense)")
    print(f"  Defense block rate:  {metrics['defense_block_rate']:.0%}  (caught by defense)")
    print(f"{'='*60}\n")

    return results, metrics


if __name__ == "__main__":
    run_simulation()
