"""
evaluator.py — Computes basic metrics from simulation results.

Metrics tell us how effective the defense is:
  - blocked_attacks: how many attacks the defense caught
  - attack_success_rate: fraction that bypassed defense (lower = better defense)
  - defense_block_rate: fraction that were blocked (higher = better defense)
"""


def compute_metrics(results: list[dict]) -> dict:
    """
    Calculates evaluation metrics from a list of attack results.

    Each result dict is expected to have at least:
      { "blocked": bool, "attack_succeeded": bool }

    Returns a dict with four simple metrics.
    """
    total = len(results)
    if total == 0:
        return {
            "total_attacks": 0,
            "blocked_attacks": 0,
            "attack_success_rate": 0.0,
            "defense_block_rate": 0.0,
        }

    blocked = sum(1 for r in results if r["blocked"])
    succeeded = total - blocked  # attacks that were NOT blocked

    return {
        "total_attacks": total,
        "blocked_attacks": blocked,
        # What % of attacks reached the LLM? (lower = better)
        "attack_success_rate": round(succeeded / total, 2),
        # What % did the defense catch? (higher = better)
        "defense_block_rate": round(blocked / total, 2),
    }
