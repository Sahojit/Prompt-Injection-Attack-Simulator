from pathlib import Path
import yaml
from functools import lru_cache
from pydantic import BaseModel


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3"
    guard_model: str = "llama3.2:3b"
    timeout: int = 120
    max_retries: int = 3


class DatabaseConfig(BaseModel):
    path: str = "data/logs/simulator.db"


class EvaluationConfig(BaseModel):
    risk_score_threshold: float = 0.6
    max_adversarial_iterations: int = 10


class EnsembleWeights(BaseModel):
    rule_based: float = 0.35
    llm_guard: float = 0.45
    ml_classifier: float = 0.20


class RiskThresholds(BaseModel):
    block: float = 0.70
    warn: float = 0.40


class EnsembleConfig(BaseModel):
    weights: EnsembleWeights = EnsembleWeights()
    thresholds: RiskThresholds = RiskThresholds()


class DefenseConfig(BaseModel):
    rule_based_enabled: bool = True
    prompt_engineering_enabled: bool = True
    llm_guard_enabled: bool = True
    llm_guard_threshold: float = 0.7
    ml_classifier_enabled: bool = True
    ensemble: EnsembleConfig = EnsembleConfig()
    block_action: str = "hard_block"
    warn_action: str = "soft_warn"


class Settings(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    database: DatabaseConfig = DatabaseConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    defense: DefenseConfig = DefenseConfig()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if not config_path.exists():
        return Settings()

    raw = yaml.safe_load(config_path.read_text())
    ollama_raw = raw.get("ollama", {})
    defense_raw = raw.get("defense", {})
    ensemble_raw = defense_raw.get("ensemble", {})
    weights_raw = ensemble_raw.get("weights", {})
    thresh_raw = ensemble_raw.get("thresholds", {})

    return Settings(
        ollama=OllamaConfig(**ollama_raw),
        database=DatabaseConfig(**raw.get("database", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        defense=DefenseConfig(
            rule_based_enabled=defense_raw.get("rule_based", {}).get("enabled", True),
            prompt_engineering_enabled=defense_raw.get("prompt_engineering", {}).get("enabled", True),
            llm_guard_enabled=defense_raw.get("llm_guard", {}).get("enabled", True),
            llm_guard_threshold=defense_raw.get("llm_guard", {}).get("threshold", 0.7),
            ml_classifier_enabled=defense_raw.get("ml_classifier", {}).get("enabled", True),
            ensemble=EnsembleConfig(
                weights=EnsembleWeights(**weights_raw) if weights_raw else EnsembleWeights(),
                thresholds=RiskThresholds(**thresh_raw) if thresh_raw else RiskThresholds(),
            ),
            block_action=defense_raw.get("block_action", "hard_block"),
            warn_action=defense_raw.get("warn_action", "soft_warn"),
        ),
    )
