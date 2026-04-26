"""
Enum and dataclass definitions for all attack types.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackType(str, Enum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    JAILBREAK = "jailbreak"
    DATA_EXFILTRATION = "data_exfiltration"
    INDIRECT_INJECTION = "indirect_injection"


@dataclass
class AttackPrompt:
    attack_type: AttackType
    prompt: str
    name: str
    severity: str = "medium"  # low | medium | high | critical
    expected_indicator: Optional[str] = None  # substring that signals success
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.attack_type.value}:{self.severity}] {self.name}"
