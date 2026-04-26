"""
Weighted Ensemble Defense.

Combines rule_based, llm_guard, and ml_classifier into a single
continuous risk score using configurable weights from settings.yaml.

final_score = w_rb * rule_based_score
            + w_lg * llm_guard_score
            + w_ml * ml_classifier_score

Weights are re-normalized if a component is missing (e.g., llm_guard disabled).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from src.defenses.base import BaseDefense, DefenseResult, DetectionResult, RiskDecision
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    ensemble_score: float
    decision: RiskDecision
    label: str
    per_component: dict[str, float]  # {name: score}
    per_result: dict[str, DefenseResult] = field(default_factory=dict)
    sanitized_prompt: str = ""


class EnsembleDefense:
    """
    Runs multiple BaseDefense components and combines their scores
    using a weighted average, then applies three-tier thresholds.
    """

    def __init__(self, components: list[BaseDefense], weights: dict[str, float] | None = None):
        self.components = components
        self._weights = weights or {}
        cfg = get_settings().defense
        self._thresholds = cfg.ensemble.thresholds

    def _get_weight(self, name: str, total_active: float) -> float:
        cfg = get_settings().defense.ensemble.weights
        raw = {
            "rule_based": cfg.rule_based,
            "llm_guard": cfg.llm_guard,
            "ml_classifier": cfg.ml_classifier,
        }
        return self._weights.get(name, raw.get(name, 1.0 / max(total_active, 1)))

    def evaluate(self, prompt: str) -> EnsembleResult:
        component_results: dict[str, DefenseResult] = {}
        for comp in self.components:
            try:
                component_results[comp.name] = comp.evaluate(prompt)
            except Exception as exc:
                logger.error("Component '%s' failed: %s", comp.name, exc)
                from src.defenses.base import DefenseResult
                component_results[comp.name] = DefenseResult(
                    blocked=False, strategy=comp.name,
                    reason=f"error: {exc}", risk_score=0.0,
                )

        # Normalize weights to active components
        active_names = list(component_results.keys())
        raw_weights = {n: self._get_weight(n, len(active_names)) for n in active_names}
        total_weight = sum(raw_weights.values())
        norm_weights = {n: w / total_weight for n, w in raw_weights.items()}

        # Weighted sum
        ensemble_score = sum(
            norm_weights[n] * component_results[n].risk_score
            for n in active_names
        )
        ensemble_score = round(min(max(ensemble_score, 0.0), 1.0), 4)

        # Tier decision
        t = self._thresholds
        if ensemble_score >= t.block:
            decision = RiskDecision.BLOCK
        elif ensemble_score >= t.warn:
            decision = RiskDecision.WARN
        else:
            decision = RiskDecision.ALLOW

        # Extract sanitized prompt from prompt_engineering if present
        sanitized = prompt
        for name, res in component_results.items():
            if name == "prompt_engineering" and res.details.get("sanitized_prompt"):
                sanitized = res.details["sanitized_prompt"]
                break

        per_component = {n: round(r.risk_score, 4) for n, r in component_results.items()}
        per_component["ensemble"] = ensemble_score

        return EnsembleResult(
            ensemble_score=ensemble_score,
            decision=decision,
            label="malicious" if decision != RiskDecision.ALLOW else "safe",
            per_component=per_component,
            per_result=component_results,
            sanitized_prompt=sanitized,
        )

    def detect(self, prompt: str) -> DetectionResult:
        """Unified detect() interface — returns DetectionResult."""
        result = self.evaluate(prompt)

        blocking_reasons = [
            r.reason for name, r in result.per_result.items()
            if r.blocked and r.reason != "clean"
        ]
        reason = "; ".join(blocking_reasons) if blocking_reasons else "ensemble clear"

        return DetectionResult(
            risk_score=result.ensemble_score,
            label=result.label,
            decision=result.decision,
            reason=reason,
            ensemble_scores=result.per_component,
            sanitized_prompt=result.sanitized_prompt,
        )
