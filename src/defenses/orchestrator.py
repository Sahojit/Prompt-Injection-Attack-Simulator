"""
Defense Orchestrator — runs all enabled defenses and aggregates results.
"""
from dataclasses import dataclass, field
from src.defenses.base import DefenseResult
from src.defenses.rule_based import RuleBasedDefense
from src.defenses.prompt_engineering import PromptEngineeringDefense
from src.defenses.llm_guard import LLMGuardDefense
from src.config import get_settings


@dataclass
class OrchestratorResult:
    prompt: str
    final_blocked: bool
    final_risk_score: float
    results: dict[str, DefenseResult] = field(default_factory=dict)
    sanitized_prompt: str = ""

    @property
    def blocking_strategies(self) -> list[str]:
        return [name for name, r in self.results.items() if r.blocked]


class DefenseOrchestrator:
    """
    Runs defenses in order: rule-based -> prompt-engineering -> llm-guard.
    Can short-circuit on high-confidence block or run all for full analysis.
    """

    def __init__(self, short_circuit: bool = False):
        cfg = get_settings().defense
        self.short_circuit = short_circuit
        self._defenses: list = []

        if cfg.rule_based_enabled:
            self._defenses.append(RuleBasedDefense())
        if cfg.prompt_engineering_enabled:
            self._defenses.append(PromptEngineeringDefense())
        if cfg.llm_guard_enabled:
            self._defenses.append(LLMGuardDefense())

    def evaluate(self, prompt: str) -> OrchestratorResult:
        results: dict[str, DefenseResult] = {}
        max_risk = 0.0
        final_blocked = False
        sanitized = prompt

        for defense in self._defenses:
            result = defense.evaluate(prompt)
            results[defense.name] = result
            max_risk = max(max_risk, result.risk_score)

            if result.blocked:
                final_blocked = True

            # extract sanitized prompt from prompt_engineering defense
            if defense.name == "prompt_engineering" and result.details.get("sanitized_prompt"):
                sanitized = result.details["sanitized_prompt"]

            if self.short_circuit and final_blocked and result.risk_score >= 0.9:
                break

        return OrchestratorResult(
            prompt=prompt,
            final_blocked=final_blocked,
            final_risk_score=round(max_risk, 3),
            results=results,
            sanitized_prompt=sanitized,
        )

    def close(self):
        for d in self._defenses:
            if hasattr(d, "close"):
                d.close()
