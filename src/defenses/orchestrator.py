"""
Defense Orchestrator v2 — full defense pipeline.

Flow:
  1. Run all enabled components individually
  2. Combine via EnsembleDefense (weighted average → risk score)
  3. Map score to RiskDecision tier (BLOCK / WARN / ALLOW)
  4. Apply tier-specific action (hard_block / soft_warn / rewrite / safe_replace)
  5. Expose unified detect() interface

Backward-compatible: OrchestratorResult is retained with new fields added.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from src.defenses.base import DefenseResult, DetectionResult, RiskDecision
from src.defenses.rule_based import RuleBasedDefense
from src.defenses.prompt_engineering import PromptEngineeringDefense
from src.defenses.llm_guard import LLMGuardDefense
from src.defenses.ml_classifier import MLClassifierDefense
from src.defenses.ensemble import EnsembleDefense, EnsembleResult
from src.defenses.actions import apply_action, ActionResult
from src.config import get_settings


@dataclass
class OrchestratorResult:
    """Backward-compatible result object, extended with ensemble + action fields."""
    prompt: str
    final_blocked: bool
    final_risk_score: float
    results: dict[str, DefenseResult] = field(default_factory=dict)
    sanitized_prompt: str = ""

    # v2 fields
    risk_decision: RiskDecision = RiskDecision.ALLOW
    ensemble_scores: dict = field(default_factory=dict)
    action: ActionResult = None
    detection: DetectionResult = None

    @property
    def blocking_strategies(self) -> list[str]:
        return [name for name, r in self.results.items() if r.blocked]

    @property
    def risk_tier(self) -> str:
        return self.risk_decision.value


class DefenseOrchestrator:
    """
    Unified defense pipeline with ensemble scoring, risk tiers, and defense actions.

    Args:
        defense_mode: "full" (all defenses) | "rule_only" | "no_llm" (skip llm_guard)
        short_circuit: stop early on high-confidence block
    """

    def __init__(self, defense_mode: str = "full", short_circuit: bool = False):
        cfg = get_settings().defense
        self.short_circuit = short_circuit
        self.defense_mode = defense_mode
        self._components: list = []
        self._pe_defense = None

        if cfg.rule_based_enabled:
            rb = RuleBasedDefense()
            self._components.append(rb)

        if cfg.ml_classifier_enabled and defense_mode in ("full", "no_llm", "rule_only"):
            self._components.append(MLClassifierDefense())

        if cfg.llm_guard_enabled and defense_mode == "full":
            self._components.append(LLMGuardDefense())

        if cfg.prompt_engineering_enabled:
            self._pe_defense = PromptEngineeringDefense()

        self._ensemble = EnsembleDefense(self._components)
        self._cfg = cfg

    def detect(self, prompt: str) -> DetectionResult:
        """
        Primary public interface.
        Returns DetectionResult with risk_score, label, decision, and action_taken.
        """
        result = self.evaluate(prompt)
        return result.detection

    def evaluate(self, prompt: str) -> OrchestratorResult:
        """
        Full pipeline evaluation. Returns OrchestratorResult for backward compatibility
        while populating all v2 fields.
        """
        # 1. Prompt engineering sanitization (always runs, doesn't block on its own)
        sanitized = prompt
        pe_result = None
        if self._pe_defense:
            pe_result = self._pe_defense.evaluate(prompt)
            sanitized = pe_result.details.get("sanitized_prompt", prompt)

        # 2. Ensemble scoring
        ensemble_result: EnsembleResult = self._ensemble.evaluate(prompt)

        # Update sanitized from ensemble (in case prompt_engineering is a component too)
        if ensemble_result.sanitized_prompt != prompt:
            sanitized = ensemble_result.sanitized_prompt

        # 3. Build per-strategy DefenseResult dict (for legacy compatibility)
        all_results: dict[str, DefenseResult] = dict(ensemble_result.per_result)
        if pe_result:
            all_results["prompt_engineering"] = pe_result

        # 4. Apply defense action based on risk tier
        cfg = self._cfg
        if ensemble_result.decision == RiskDecision.BLOCK:
            action_name = cfg.block_action
        elif ensemble_result.decision == RiskDecision.WARN:
            action_name = cfg.warn_action
        else:
            action_name = "none"

        blocking_reasons = [
            r.reason for r in all_results.values()
            if r.blocked and r.reason != "clean"
        ]
        reason = "; ".join(blocking_reasons) if blocking_reasons else "ensemble clear"

        action = apply_action(
            action_name=action_name,
            original_prompt=prompt,
            sanitized_prompt=sanitized,
            risk_score=ensemble_result.ensemble_score,
            reason=reason,
        )

        detection = DetectionResult(
            risk_score=ensemble_result.ensemble_score,
            label=ensemble_result.label,
            decision=ensemble_result.decision,
            reason=reason,
            ensemble_scores=ensemble_result.per_component,
            sanitized_prompt=sanitized,
            action_taken=action.action.value,
        )

        return OrchestratorResult(
            prompt=prompt,
            final_blocked=(ensemble_result.decision == RiskDecision.BLOCK),
            final_risk_score=ensemble_result.ensemble_score,
            results=all_results,
            sanitized_prompt=sanitized,
            risk_decision=ensemble_result.decision,
            ensemble_scores=ensemble_result.per_component,
            action=action,
            detection=detection,
        )

    def close(self):
        for comp in self._components:
            if hasattr(comp, "close"):
                comp.close()
