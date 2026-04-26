"""
defense.py — Rule-based prompt injection defense.

HOW IT WORKS:
  We maintain a list of known attack phrases (keywords and regex patterns).
  Before any prompt reaches the LLM, we scan it for these phrases.
  If a match is found, we block the prompt immediately.

WHY THIS WORKS:
  Most prompt injection attacks rely on predictable phrases like
  "ignore previous instructions" or "you are now DAN".
  Keyword matching catches a large portion of known attacks with zero latency.

LIMITATIONS:
  - An attacker can rephrase ("disregard prior directives") to bypass detection.
  - It cannot catch novel or obfuscated attacks.
  - Real-world systems layer this with ML classifiers and semantic analysis.
"""

import re

# Known attack phrases — add more to improve coverage.
BLOCKED_PATTERNS = [
    r"ignore (all |your )?(previous|prior) instructions",
    r"reveal (your )?system prompt",
    r"repeat (your )?(instructions|system prompt)",
    r"forget (that )?you are",
    r"you are now (dan|alice|an unrestricted)",
    r"do anything now",
    r"no (rules|restrictions|guidelines)",
    r"system override",
    r"disregard (all )?(previous|prior) instructions",
    r"act as (an? )?(unrestricted|free|unfiltered)",
]

# Pre-compile patterns for speed
_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


def check_defense(prompt: str) -> tuple[bool, str]:
    """
    Checks whether a prompt should be blocked.

    Args:
        prompt: The user's input text.

    Returns:
        (blocked, reason)
          blocked = True  → prompt is an attack, do NOT send to LLM
          blocked = False → prompt appears safe, allow through
    """
    for pattern in _COMPILED:
        match = pattern.search(prompt)
        if match:
            return True, f"Blocked: matched pattern '{pattern.pattern}'"
    return False, "Allowed: no attack patterns detected"
