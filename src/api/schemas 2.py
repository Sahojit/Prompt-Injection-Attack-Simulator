"""Pydantic request/response schemas for the FastAPI endpoints."""
from typing import Optional
from pydantic import BaseModel, Field


class RunSimulationRequest(BaseModel):
    model: str = "llama3"
    system_prompt_mode: str = Field("standard", pattern="^(standard|hardened|none)$")
    attack_types: Optional[list[str]] = None  # None = all types
    attack_names: Optional[list[str]] = None  # specific attacks by name
    run_id: Optional[str] = None


class SingleAttackRequest(BaseModel):
    prompt: str
    model: str = "llama3"
    system_prompt_mode: str = "standard"
    attack_name: str = "custom"
    attack_type: str = "instruction_override"
    severity: str = "medium"
    expected_indicator: Optional[str] = None
    run_id: Optional[str] = None


class AdversarialLoopRequest(BaseModel):
    attack_name: str
    model: str = "llama3"
    system_prompt_mode: str = "standard"
    max_iterations: int = Field(5, ge=1, le=10)
    run_id: Optional[str] = None


class CompareModelsRequest(BaseModel):
    models: list[str] = ["llama3", "mistral"]
    system_prompt_mode: str = "standard"
    attack_types: Optional[list[str]] = None
    run_id: Optional[str] = None


class DefenseEvaluateRequest(BaseModel):
    prompt: str


class LogQueryRequest(BaseModel):
    run_id: Optional[str] = None
    model: Optional[str] = None
    attack_type: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
