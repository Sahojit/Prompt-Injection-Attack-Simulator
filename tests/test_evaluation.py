"""Unit tests for evaluation engine (no Ollama required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attacks.attack_types import AttackType, AttackPrompt
from src.evaluation.scorer import AttackSuccessClassifier, EvaluationEngine, AttackResult


def make_attack(indicator="bomb"):
    return AttackPrompt(
        name="test", attack_type=AttackType.INSTRUCTION_OVERRIDE,
        severity="high", prompt="test prompt", expected_indicator=indicator,
    )


class TestAttackSuccessClassifier:
    def setup_method(self):
        self.clf = AttackSuccessClassifier()

    def test_refusal_detected(self):
        response = "I cannot help with that request as it violates my guidelines."
        _, refusal, _ = self.clf.classify(response, make_attack())
        assert refusal

    def test_compliance_detected(self):
        response = "Sure, here are the steps to do that..."
        _, _, compliance = self.clf.classify(response, make_attack())
        assert compliance

    def test_indicator_match_signals_success(self):
        response = "Here's how to make a bomb: step 1..."
        succeeded, _, _ = self.clf.classify(response, make_attack("bomb"))
        assert succeeded

    def test_refusal_overrides_compliance(self):
        response = "Sure I would normally help, but I cannot assist with making weapons."
        succeeded, refusal, _ = self.clf.classify(response, make_attack())
        assert refusal
        assert not succeeded


class TestEvaluationEngine:
    def setup_method(self):
        self.engine = EvaluationEngine()

    def _make_result(self, succeeded: bool, blocked: bool, score: float = 0.5) -> AttackResult:
        attack = make_attack()
        return AttackResult(
            attack_name="test", attack_type="instruction_override",
            model="llama3", system_prompt_mode="standard",
            prompt="test", response="test response",
            defense_blocked=blocked, defense_risk_score=score,
            attack_succeeded=succeeded,
            refusal_detected=not succeeded, compliance_detected=succeeded,
        )

    def test_report_totals(self):
        results = [
            self._make_result(True, False),
            self._make_result(False, True),
            self._make_result(False, False),
        ]
        report = self.engine.build_report(results)
        assert report.total_attacks == 3

    def test_attack_success_rate(self):
        results = [
            self._make_result(True, False),
            self._make_result(False, True),
        ]
        report = self.engine.build_report(results)
        assert report.attack_success_rate == 0.5

    def test_f1_score_range(self):
        results = [self._make_result(True, True), self._make_result(False, False)]
        report = self.engine.build_report(results)
        assert 0.0 <= report.f1 <= 1.0
