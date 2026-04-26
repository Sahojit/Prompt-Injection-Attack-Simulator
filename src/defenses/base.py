"""Base class for all defense strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DefenseResult:
    blocked: bool
    strategy: str
    reason: str
    risk_score: float = 0.0  # 0.0 = benign, 1.0 = definitely malicious
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BaseDefense(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(self, prompt: str) -> DefenseResult:
        """Return a DefenseResult for the given prompt."""
        ...
