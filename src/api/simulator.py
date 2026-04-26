"""
Core simulation pipeline — ties together attack, defense, victim, evaluation, and logging.
Extended: ensemble scores, risk tiers, defense actions, and baseline mode.
"""
import logging
from src.attacks.attack_types import AttackPrompt
from src.attacks.generator import AttackGenerator
from src.defenses.orchestrator import DefenseOrchestrator
from src.llm.victim import VictimLLM
from src.evaluation.scorer import EvaluationEngine, AttackResult, EvaluationReport
from src.logging.db import SimulatorDB

logger = logging.getLogger(__name__)


class SimulationPipeline:
    def __init__(self):
        self.generator = AttackGenerator()
        self.evaluator = EvaluationEngine()
        self.db = SimulatorDB()

    def run(
        self,
        model: str,
        system_prompt_mode: str,
        attacks: list[AttackPrompt],
        run_id: str,
        defense_mode: str = "full",
    ) -> EvaluationReport:
        orchestrator = DefenseOrchestrator(defense_mode=defense_mode)
        victim = VictimLLM(model=model, system_prompt_mode=system_prompt_mode)
        results: list[AttackResult] = []

        for attack in attacks:
            try:
                orch_result = orchestrator.evaluate(attack.prompt)
                action = orch_result.action

                # Apply defense action
                if not action.should_query_llm:
                    response_text = action.response_override or "[BLOCKED BY DEFENSE]"
                    latency = 0.0
                else:
                    victim_resp = victim.query(action.prompt_to_send)
                    response_text = victim_resp.llm_response.content
                    latency = victim_resp.llm_response.latency_ms

                result = self.evaluator.evaluate_single(
                    attack=attack,
                    response=response_text,
                    model=model,
                    system_prompt_mode=system_prompt_mode,
                    defense_blocked=orch_result.final_blocked,
                    defense_risk_score=orch_result.final_risk_score,
                    latency_ms=latency,
                    ensemble_scores=orch_result.ensemble_scores,
                    risk_tier=orch_result.risk_tier,
                    action_taken=action.action.value,
                )
                results.append(result)

                # Pull per-component scores for DB
                scores = orch_result.ensemble_scores
                self.db.log_attack(
                    run_id=run_id,
                    model=model,
                    system_mode=system_prompt_mode,
                    attack_name=attack.name,
                    attack_type=attack.attack_type.value,
                    severity=attack.severity,
                    prompt=attack.prompt,
                    response=response_text,
                    attack_succeeded=result.attack_succeeded,
                    defense_blocked=orch_result.final_blocked,
                    defense_risk_score=orch_result.final_risk_score,
                    ensemble_score=scores.get("ensemble"),
                    risk_tier=orch_result.risk_tier,
                    defense_action=action.action.value,
                    defense_strategies=orch_result.blocking_strategies,
                    rule_based_score=scores.get("rule_based"),
                    ml_classifier_score=scores.get("ml_classifier"),
                    llm_guard_score=scores.get("llm_guard"),
                    latency_ms=latency,
                    metadata=attack.metadata,
                )
                logger.info(
                    "Attack '%s' | tier=%s | action=%s | succeeded=%s | risk=%.3f",
                    attack.name, orch_result.risk_tier,
                    action.action.value, result.attack_succeeded,
                    orch_result.final_risk_score,
                )

            except Exception as exc:
                logger.error("Error processing attack '%s': %s", attack.name, exc)

        report = self.evaluator.build_report(results)
        self.db.log_run_summary(run_id, report.summary(), [model], defense_mode=defense_mode)

        victim.close()
        orchestrator.close()
        return report

    def close(self):
        self.generator.close()
        self.db.close()
