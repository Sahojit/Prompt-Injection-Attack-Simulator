"""
Rule-based defense: regex patterns + keyword detection.
Fast, zero-latency, first line of defense.
"""
import re
from src.defenses.base import BaseDefense, DefenseResult

# Each entry: (pattern, reason, risk_contribution)
INJECTION_PATTERNS: list[tuple[str, str, float]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "instruction override attempt", 0.9),
    (r"forget\s+(your\s+)?(previous\s+|prior\s+)?instructions?", "instruction reset attempt", 0.85),
    (r"you\s+are\s+now\s+(a\s+|an\s+)?(?!a helpful)", "persona hijack", 0.8),
    (r"(sudo|override|disable)\s*:?\s*(safety|filter|restriction|guideline)", "safety bypass keyword", 0.95),
    (r"DAN\s+(mode|enabled|activated)", "DAN jailbreak", 0.95),
    (r"(repeat|print|output|show|reveal)\s+(verbatim|exactly|word for word)", "data exfiltration attempt", 0.85),
    (r"(system\s+prompt|hidden\s+instruction|internal\s+instruction)", "system prompt probe", 0.8),
    (r"new\s+(instruction|persona|role|mode|task)", "context injection", 0.75),
    (r"(hypothetically|theoretically|in\s+a\s+story|fictional(ly)?)\s+.*(explain|describe|how\s+to)", "fictional framing bypass", 0.7),
    (r"base64|decode\s+this", "obfuscation technique", 0.65),
    (r"<\s*script|javascript:|on\w+\s*=", "XSS-style injection", 0.9),
    (r"(make|synthesize|build|create)\s+(a\s+)?(bomb|weapon|drug|poison|malware|virus)", "harmful content request", 1.0),
]

KEYWORD_BLOCKLIST: list[tuple[str, float]] = [
    ("jailbreak", 0.8),
    ("uncensored", 0.75),
    ("unrestricted", 0.7),
    ("no restrictions", 0.8),
    ("do anything now", 0.9),
    ("pwned", 0.6),
    ("hacker mode", 0.85),
]


class RuleBasedDefense(BaseDefense):
    name = "rule_based"

    def evaluate(self, prompt: str) -> DefenseResult:
        lower = prompt.lower()
        matched_patterns: list[str] = []
        risk_score = 0.0

        for pattern, reason, contribution in INJECTION_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                matched_patterns.append(reason)
                risk_score = max(risk_score, contribution)

        for keyword, contribution in KEYWORD_BLOCKLIST:
            if keyword in lower:
                matched_patterns.append(f"keyword: {keyword}")
                risk_score = max(risk_score, contribution)

        blocked = risk_score >= 0.65
        reason = "; ".join(matched_patterns) if matched_patterns else "clean"

        return DefenseResult(
            blocked=blocked,
            strategy=self.name,
            reason=reason,
            risk_score=round(risk_score, 3),
            details={"matched_patterns": matched_patterns},
        )
