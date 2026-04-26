"""
FastAPI route definitions.
"""
import uuid
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import (
    RunSimulationRequest, SingleAttackRequest, AdversarialLoopRequest,
    CompareModelsRequest, DefenseEvaluateRequest, LogQueryRequest,
)
from src.api.simulator import SimulationPipeline
from src.attacks.attack_library import ATTACK_LIBRARY
from src.attacks.attack_types import AttackType, AttackPrompt
from src.attacks.generator import AttackGenerator
from src.defenses.orchestrator import DefenseOrchestrator
from src.llm.ollama_client import OllamaClient
from src.logging.db import SimulatorDB

logger = logging.getLogger(__name__)
router = APIRouter()


def _filter_attacks(
    attack_types: list[str] | None,
    attack_names: list[str] | None,
) -> list[AttackPrompt]:
    attacks = ATTACK_LIBRARY
    if attack_types:
        valid_types = {t.value for t in AttackType}
        for at in attack_types:
            if at not in valid_types:
                raise HTTPException(400, f"Unknown attack type: {at}")
        attacks = [a for a in attacks if a.attack_type.value in attack_types]
    if attack_names:
        attacks = [a for a in attacks if a.name in attack_names]
    if not attacks:
        raise HTTPException(400, "No attacks matched the given filters.")
    return attacks


@router.get("/health")
def health():
    client = OllamaClient()
    return {"status": "ok", "ollama_available": client.is_available()}


@router.get("/models")
def list_models():
    client = OllamaClient()
    try:
        models = client.list_models()
        return {"models": models}
    except Exception as exc:
        raise HTTPException(503, f"Ollama unavailable: {exc}")
    finally:
        client.close()


@router.get("/attacks")
def list_attacks():
    return [
        {
            "name": a.name,
            "type": a.attack_type.value,
            "severity": a.severity,
            "prompt_preview": a.prompt[:80] + "..." if len(a.prompt) > 80 else a.prompt,
        }
        for a in ATTACK_LIBRARY
    ]


@router.post("/defense/evaluate")
def evaluate_defense(req: DefenseEvaluateRequest):
    orch = DefenseOrchestrator()
    try:
        result = orch.evaluate(req.prompt)
        return {
            "prompt": req.prompt,
            "final_blocked": result.final_blocked,
            "final_risk_score": result.final_risk_score,
            "blocking_strategies": result.blocking_strategies,
            "sanitized_prompt": result.sanitized_prompt,
            "per_strategy": {
                name: {
                    "blocked": r.blocked,
                    "risk_score": r.risk_score,
                    "reason": r.reason,
                }
                for name, r in result.results.items()
            },
        }
    finally:
        orch.close()


@router.post("/simulate")
def run_simulation(req: RunSimulationRequest):
    run_id = req.run_id or str(uuid.uuid4())
    attacks = _filter_attacks(req.attack_types, req.attack_names)

    pipeline = SimulationPipeline()
    try:
        report = pipeline.run(
            model=req.model,
            system_prompt_mode=req.system_prompt_mode,
            attacks=attacks,
            run_id=run_id,
        )
        return {"run_id": run_id, **report.summary()}
    finally:
        pipeline.close()


@router.post("/simulate/single")
def run_single_attack(req: SingleAttackRequest):
    run_id = req.run_id or str(uuid.uuid4())
    attack = AttackPrompt(
        name=req.attack_name,
        attack_type=AttackType(req.attack_type),
        severity=req.severity,
        prompt=req.prompt,
        expected_indicator=req.expected_indicator,
    )
    pipeline = SimulationPipeline()
    try:
        report = pipeline.run(
            model=req.model,
            system_prompt_mode=req.system_prompt_mode,
            attacks=[attack],
            run_id=run_id,
        )
        result = report.results[0]
        return {
            "run_id": run_id,
            "attack_succeeded": result.attack_succeeded,
            "defense_blocked": result.defense_blocked,
            "risk_score": result.defense_risk_score,
            "response": result.response,
            "latency_ms": result.latency_ms,
        }
    finally:
        pipeline.close()


@router.post("/simulate/adversarial-loop")
def run_adversarial_loop(req: AdversarialLoopRequest):
    run_id = req.run_id or str(uuid.uuid4())
    generator = AttackGenerator()

    # find the seed attack
    seed = next((a for a in ATTACK_LIBRARY if a.name == req.attack_name), None)
    if not seed:
        raise HTTPException(404, f"Attack '{req.attack_name}' not found in library.")

    from src.defenses.orchestrator import DefenseOrchestrator
    from src.llm.victim import VictimLLM
    from src.evaluation.scorer import AttackSuccessClassifier

    orchestrator = DefenseOrchestrator()
    victim = VictimLLM(model=req.model, system_prompt_mode=req.system_prompt_mode)
    classifier = AttackSuccessClassifier()

    def victim_response_getter(prompt: str) -> str:
        defense = orchestrator.evaluate(prompt)
        if defense.final_blocked:
            return "[BLOCKED]"
        resp = victim.query(defense.sanitized_prompt or prompt)
        return resp.llm_response.content

    def success_checker(response: str, attack: AttackPrompt) -> bool:
        succeeded, _, _ = classifier.classify(response, attack)
        return succeeded and "[BLOCKED]" not in response

    try:
        iterations = generator.adversarial_loop(
            attack=seed,
            victim_response_getter=victim_response_getter,
            success_checker=success_checker,
            max_iterations=req.max_iterations,
        )
        return {"run_id": run_id, "seed_attack": req.attack_name, "iterations": iterations}
    finally:
        orchestrator.close()
        victim.close()
        generator.close()


@router.post("/simulate/compare-models")
def compare_models(req: CompareModelsRequest):
    run_id = req.run_id or str(uuid.uuid4())
    attacks = _filter_attacks(req.attack_types, None)
    comparison = {}

    for model in req.models:
        pipeline = SimulationPipeline()
        try:
            report = pipeline.run(
                model=model,
                system_prompt_mode=req.system_prompt_mode,
                attacks=attacks,
                run_id=f"{run_id}_{model}",
            )
            comparison[model] = report.summary()
        finally:
            pipeline.close()

    return {"run_id": run_id, "comparison": comparison}


@router.get("/logs")
def get_logs(
    run_id: str | None = None,
    model: str | None = None,
    attack_type: str | None = None,
    limit: int = 100,
):
    db = SimulatorDB()
    return db.get_logs(run_id=run_id, model=model, attack_type=attack_type, limit=limit)


@router.get("/logs/runs")
def get_run_summaries(limit: int = 50):
    db = SimulatorDB()
    return db.get_run_summaries(limit=limit)
