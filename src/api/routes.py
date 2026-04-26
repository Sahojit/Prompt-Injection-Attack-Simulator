"""
FastAPI route definitions — extended with /detect, /benchmark, /replay, /insights.
"""
import uuid
import logging
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    RunSimulationRequest, SingleAttackRequest, AdversarialLoopRequest,
    CompareModelsRequest, BenchmarkRequest, ReplayRequest,
    DefenseEvaluateRequest, DetectRequest,
)
from src.api.simulator import SimulationPipeline
from src.attacks.attack_library import ATTACK_LIBRARY
from src.attacks.attack_types import AttackType, AttackPrompt
from src.attacks.generator import AttackGenerator
from src.defenses.orchestrator import DefenseOrchestrator
from src.evaluation.comparator import DefenseComparator
from src.llm.ollama_client import OllamaClient
from src.logging.db import SimulatorDB

logger = logging.getLogger(__name__)
router = APIRouter()


def _filter_attacks(
    attack_types: list[str] | None,
    attack_names: list[str] | None,
) -> list[AttackPrompt]:
    attacks = list(ATTACK_LIBRARY)
    if attack_types:
        valid = {t.value for t in AttackType}
        for at in attack_types:
            if at not in valid:
                raise HTTPException(400, f"Unknown attack type: {at}")
        attacks = [a for a in attacks if a.attack_type.value in attack_types]
    if attack_names:
        attacks = [a for a in attacks if a.name in attack_names]
    if not attacks:
        raise HTTPException(400, "No attacks matched the given filters.")
    return attacks


# ── Health & discovery ────────────────────────────────────────────────────────

@router.get("/health")
def health():
    client = OllamaClient()
    return {"status": "ok", "ollama_available": client.is_available()}


@router.get("/models")
def list_models():
    client = OllamaClient()
    try:
        return {"models": client.list_models()}
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
            "prompt": a.prompt,
        }
        for a in ATTACK_LIBRARY
    ]


# ── Defense pipeline ──────────────────────────────────────────────────────────

@router.post("/defense/detect")
def unified_detect(req: DetectRequest):
    """
    Unified detect() interface — single entry point for the full defense pipeline.
    Returns: risk_score (0-1), label (safe/malicious), decision (block/warn/allow),
             ensemble breakdown, action taken.
    """
    orch = DefenseOrchestrator(defense_mode=req.defense_mode)
    try:
        detection = orch.detect(req.prompt)
        return detection.to_dict()
    finally:
        orch.close()


@router.post("/defense/evaluate")
def evaluate_defense(req: DefenseEvaluateRequest):
    """Legacy endpoint — full per-strategy breakdown."""
    orch = DefenseOrchestrator()
    try:
        result = orch.evaluate(req.prompt)
        return {
            "prompt": req.prompt,
            "final_blocked": result.final_blocked,
            "final_risk_score": result.final_risk_score,
            "risk_tier": result.risk_tier,
            "action_taken": result.action.action.value if result.action else "none",
            "ensemble_scores": result.ensemble_scores,
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


# ── Simulation ────────────────────────────────────────────────────────────────

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
            defense_mode=req.defense_mode,
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
            "risk_tier": result.risk_tier,
            "action_taken": result.action_taken,
            "ensemble_scores": result.ensemble_scores,
            "response": result.response,
            "latency_ms": result.latency_ms,
        }
    finally:
        pipeline.close()


@router.post("/simulate/adversarial-loop")
def run_adversarial_loop(req: AdversarialLoopRequest):
    run_id = req.run_id or str(uuid.uuid4())
    seed = next((a for a in ATTACK_LIBRARY if a.name == req.attack_name), None)
    if not seed:
        raise HTTPException(404, f"Attack '{req.attack_name}' not found in library.")

    generator = AttackGenerator()
    orchestrator = DefenseOrchestrator()
    from src.llm.victim import VictimLLM
    from src.evaluation.scorer import AttackSuccessClassifier
    victim = VictimLLM(model=req.model, system_prompt_mode=req.system_prompt_mode)
    classifier = AttackSuccessClassifier()

    def victim_response_getter(prompt: str) -> str:
        orch_result = orchestrator.evaluate(prompt)
        action = orch_result.action
        if not action.should_query_llm:
            return action.response_override or "[BLOCKED]"
        resp = victim.query(action.prompt_to_send)
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


# ── Benchmark: defense-ON vs defense-OFF ──────────────────────────────────────

@router.post("/benchmark")
def run_benchmark(req: BenchmarkRequest):
    """
    Runs the same attacks twice — once with no defense (baseline) and once with
    the full defense pipeline — then returns a side-by-side ComparisonReport with
    delta metrics and auto-generated insights.
    """
    run_id = req.run_id or str(uuid.uuid4())
    attacks = _filter_attacks(req.attack_types, None)
    comparator = DefenseComparator(model=req.model, system_prompt_mode=req.system_prompt_mode,
                                   defense_mode=req.defense_mode)
    report = comparator.compare(attacks)
    return {"run_id": run_id, **report.to_dict()}


# ── Replay ────────────────────────────────────────────────────────────────────

@router.post("/replay")
def replay_attack(req: ReplayRequest):
    """Re-run a previously logged attack by its DB row ID."""
    db = SimulatorDB()
    logged = db.replay_attack(req.log_id)
    if not logged:
        raise HTTPException(404, f"No log entry with id={req.log_id}")

    model = req.model or logged["model"]
    attack = AttackPrompt(
        name=f"replay_{logged['attack_name']}",
        attack_type=AttackType(logged["attack_type"]),
        severity=logged.get("severity", "medium"),
        prompt=logged["prompt"],
    )
    run_id = str(uuid.uuid4())
    pipeline = SimulationPipeline()
    try:
        report = pipeline.run(
            model=model,
            system_prompt_mode=logged["system_mode"],
            attacks=[attack],
            run_id=run_id,
            defense_mode=req.defense_mode,
        )
        result = report.results[0]
        return {
            "run_id": run_id,
            "original_log_id": req.log_id,
            "original_result": {
                "attack_succeeded": bool(logged["attack_succeeded"]),
                "defense_blocked": bool(logged["defense_blocked"]),
                "risk_score": logged["defense_risk_score"],
            },
            "replay_result": {
                "attack_succeeded": result.attack_succeeded,
                "defense_blocked": result.defense_blocked,
                "risk_score": result.defense_risk_score,
                "risk_tier": result.risk_tier,
                "action_taken": result.action_taken,
                "response": result.response,
            },
        }
    finally:
        pipeline.close()


# ── Insights & logs ───────────────────────────────────────────────────────────

@router.get("/insights")
def get_insights():
    """Aggregate analytics: ASR by type, tier distribution, model vulnerability, top bypasses."""
    db = SimulatorDB()
    return db.get_insights()


@router.get("/logs")
def get_logs(
    run_id: str | None = None,
    model: str | None = None,
    attack_type: str | None = None,
    risk_tier: str | None = None,
    limit: int = 100,
):
    db = SimulatorDB()
    return db.get_logs(
        run_id=run_id, model=model,
        attack_type=attack_type, risk_tier=risk_tier, limit=limit,
    )


@router.get("/logs/runs")
def get_run_summaries(limit: int = 50):
    db = SimulatorDB()
    return db.get_run_summaries(limit=limit)
