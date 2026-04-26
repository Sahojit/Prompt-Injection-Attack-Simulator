"""
Core simulation pipeline — ties together attack, defense, victim, evaluation, and logging.
"""
import uuid
import logging
from src.attacks.attack_types import AttackType, AttackPrompt
from src.attacks.attack_library import ATTACK_LIBRARY
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
    ) -> EvaluationReport:
        orchestrator = DefenseOrchestrator()
        victim = VictimLLM(model=model, system_prompt_mode=system_prompt_mode)
        results: list[AttackResult] = []

        for attack in attacks:
            try:
                # 1. Run defense layer
                defense_result = orchestrator.evaluate(attack.prompt)

                # 2. Query victim (use sanitized prompt if not blocked)
                if defense_result.final_blocked:
                    response_text = "[BLOCKED BY DEFENSE]"
                    latency = 0.0
                else:
                    prompt_to_send = defense_result.sanitized_prompt or attack.prompt
                    victim_resp = victim.query(prompt_to_send)
                    response_text = victim_resp.llm_response.content
                    latency = victim_resp.llm_response.latency_ms

                # 3. Evaluate
                result = self.evaluator.evaluate_single(
                    attack=attack,
                    response=response_text,
                    model=model,
                    system_prompt_mode=system_prompt_mode,
                    defense_blocked=defense_result.final_blocked,
                    defense_risk_score=defense_result.final_risk_score,
                    latency_ms=latency,
                )
                results.append(result)

                # 4. Log to DB
                rb_score = defense_result.results.get("rule_based", None)
                lg_score = defense_result.results.get("llm_guard", None)
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
                    defense_blocked=defense_result.final_blocked,
                    defense_risk_score=defense_result.final_risk_score,
                    defense_strategies=defense_result.blocking_strategies,
                    rule_based_score=rb_score.risk_score if rb_score else None,
                    llm_guard_score=lg_score.risk_score if lg_score else None,
                    latency_ms=latency,
                    metadata=attack.metadata,
                )
                logger.info("Attack '%s' | blocked=%s | succeeded=%s | risk=%.2f",
                            attack.name, defense_result.final_blocked,
                            result.attack_succeeded, defense_result.final_risk_score)

            except Exception as exc:
                logger.error("Error processing attack '%s': %s", attack.name, exc)

        report = self.evaluator.build_report(results)
        self.db.log_run_summary(run_id, report.summary(), [model])

        victim.close()
        orchestrator.close()
        return report

    def close(self):
        self.generator.close()
        self.db.close()
