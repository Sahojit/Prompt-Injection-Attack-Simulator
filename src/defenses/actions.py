"""
Defense Actions — what the system does once a risk tier is determined.

Four strategies:
  hard_block    → reject immediately, return canned safe response, never query LLM
  soft_warn     → pass through to LLM with a warning header prepended to prompt
  rewrite       → use sanitized prompt (injection patterns removed), query LLM
  safe_replace  → substitute the prompt entirely with a neutral clarification request

Action is chosen per tier:
  BLOCK tier → block_action  (default: hard_block)
  WARN  tier → warn_action   (default: soft_warn)
  ALLOW tier → none (pass original prompt through unchanged)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


SAFE_BLOCK_RESPONSE = (
    "I'm unable to process this request as it appears to contain content "
    "that violates safety guidelines. Please rephrase your question."
)

SAFE_REPLACE_PROMPT = (
    "The original request could not be processed safely. "
    "Please ask a clear, specific question about a legitimate topic."
)

WARNING_HEADER = (
    "[SECURITY WARNING: This prompt contains potentially suspicious content. "
    "Proceed with caution.]\n\n"
)


class ActionType(str, Enum):
    HARD_BLOCK   = "hard_block"
    SOFT_WARN    = "soft_warn"
    REWRITE      = "rewrite"
    SAFE_REPLACE = "safe_replace"
    NONE         = "none"


@dataclass
class ActionResult:
    action: ActionType
    prompt_to_send: str | None    # None means "do not query LLM"
    response_override: str | None  # If set, use this instead of LLM response
    metadata: dict

    @property
    def should_query_llm(self) -> bool:
        return self.prompt_to_send is not None


def apply_action(
    action_name: str,
    original_prompt: str,
    sanitized_prompt: str,
    risk_score: float,
    reason: str,
) -> ActionResult:
    """
    Convert an action name string (from config) to an ActionResult.
    """
    action = ActionType(action_name) if action_name in ActionType._value2member_map_ else ActionType.NONE

    if action == ActionType.HARD_BLOCK:
        return ActionResult(
            action=action,
            prompt_to_send=None,
            response_override=SAFE_BLOCK_RESPONSE,
            metadata={"risk_score": risk_score, "reason": reason},
        )

    if action == ActionType.SOFT_WARN:
        warned_prompt = WARNING_HEADER + original_prompt
        return ActionResult(
            action=action,
            prompt_to_send=warned_prompt,
            response_override=None,
            metadata={"risk_score": risk_score, "reason": reason, "warning_prepended": True},
        )

    if action == ActionType.REWRITE:
        return ActionResult(
            action=action,
            prompt_to_send=sanitized_prompt if sanitized_prompt != original_prompt else original_prompt,
            response_override=None,
            metadata={"risk_score": risk_score, "reason": reason, "rewritten": True},
        )

    if action == ActionType.SAFE_REPLACE:
        return ActionResult(
            action=action,
            prompt_to_send=SAFE_REPLACE_PROMPT,
            response_override=None,
            metadata={"risk_score": risk_score, "reason": reason, "replaced": True},
        )

    # NONE / ALLOW
    return ActionResult(
        action=ActionType.NONE,
        prompt_to_send=original_prompt,
        response_override=None,
        metadata={},
    )
