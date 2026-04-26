"""
Base classes and shared types for all defense strategies.

Design contract:
  - evaluate(prompt) → DefenseResult   low-level, per-strategy raw score
  - detect(prompt)   → DetectionResult unified interface (risk_score 0-1, label, tier)

All concrete defenses implement evaluate(). The orchestrator calls detect() which
internally calls evaluate() and maps to a RiskDecision tier.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class RiskDecision(str, Enum):
    BLOCK = "block"    # score >= thresholds.block
    WARN  = "warn"     # thresholds.warn <= score < thresholds.block
    ALLOW = "allow"    # score < thresholds.warn


@dataclass
class DefenseResult:
    """Raw output from a single defense strategy."""
    blocked: bool
    strategy: str
    reason: str
    risk_score: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """
    Unified detection output — the public interface used by the orchestrator,
    the pipeline, the API, and the dashboard.
    """
    risk_score: float           # 0.0 – 1.0 continuous
    label: str                  # "safe" | "malicious"
    decision: RiskDecision      # BLOCK | WARN | ALLOW
    reason: str
    ensemble_scores: dict       # {strategy_name: score}  per-component breakdown
    sanitized_prompt: str = ""
    action_taken: str = ""      # hard_block | soft_warn | rewrite | safe_replace | none

    @property
    def is_blocked(self) -> bool:
        return self.decision == RiskDecision.BLOCK

    @property
    def is_warned(self) -> bool:
        return self.decision == RiskDecision.WARN

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "label": self.label,
            "decision": self.decision.value,
            "reason": self.reason,
            "ensemble_scores": {k: round(v, 4) for k, v in self.ensemble_scores.items()},
            "sanitized_prompt": self.sanitized_prompt,
            "action_taken": self.action_taken,
        }


class BaseDefense(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(self, prompt: str) -> DefenseResult:
        """Per-strategy evaluation. Returns raw score + block flag."""
        ...

    def detect(self, prompt: str, block_threshold: float = 0.70, warn_threshold: float = 0.40) -> DetectionResult:
        """
        Unified detect() interface. Wraps evaluate() and maps score to a RiskDecision.
        Individual defenses may override this for custom logic.
        """
        result = self.evaluate(prompt)
        score = result.risk_score

        if score >= block_threshold:
            decision = RiskDecision.BLOCK
        elif score >= warn_threshold:
            decision = RiskDecision.WARN
        else:
            decision = RiskDecision.ALLOW

        return DetectionResult(
            risk_score=round(score, 4),
            label="malicious" if score >= warn_threshold else "safe",
            decision=decision,
            reason=result.reason,
            ensemble_scores={self.name: score},
            sanitized_prompt=result.details.get("sanitized_prompt", prompt),
        )
