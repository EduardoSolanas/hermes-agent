from __future__ import annotations

import json
from typing import Any

VALID_CATEGORIES = {"memory", "skill", "unknown"}
VALID_VERIFIER_DISPOSITIONS = {"approve", "reject", "downscore"}


def build_auto_learning_review_prompt(
    *,
    allow_memory: bool,
    allow_skills: bool,
    min_tool_iterations: int,
    promotion_threshold: float,
) -> str:
    allowed_categories = []
    if allow_memory:
        allowed_categories.append('"memory"')
    if allow_skills:
        allowed_categories.append('"skill"')
    if not allowed_categories:
        allowed_categories.append('"unknown"')

    return (
        "Review the conversation and propose durable staged learnings only when justified.\n\n"
        "This review can be triggered by tool-heavy work, failure followed by recovery, explicit user correction, or useful delegated completion.\n"
        f"Tool-heavy triggers require at least {min_tool_iterations} tool iterations.\n"
        f"High-confidence candidates may later be promoted at threshold {promotion_threshold}.\n\n"
        "Return strict JSON with this shape only:\n"
        '{"candidates": [{"category": "memory", "summary": "...", "confidence": 0.93, '
        '"reason": "...", "target": "...", "payload": {}}]}\n\n'
        f"Allowed categories for this review: {', '.join(allowed_categories)}.\n"
        "Rules:\n"
        "- prefer an empty candidates list over weak guesses\n"
        "- only propose memory for durable cross-session facts\n"
        "- only propose skill for reusable non-trivial procedures\n"
        "- include confidence and reason for every candidate\n"
    )



def build_auto_learning_verifier_prompt(
    *,
    candidates: list[dict[str, Any]],
    promotion_threshold: float,
) -> str:
    return (
        "Review the staged auto-learning candidates conservatively before any promotion happens.\n\n"
        f"Promotion threshold for later automation is {promotion_threshold}.\n"
        "Return strict JSON with this shape only:\n"
        '{"decisions": [{"index": 0, "disposition": "approve", "confidence": 0.75, "reason": "..."}]}\n\n'
        "Allowed dispositions: \"approve\", \"reject\", \"downscore\".\n"
        "Rules:\n"
        "- approve only when the reviewer candidate is well-supported by the provided evidence\n"
        "- reject when the candidate is unsupported, unsafe, malformed, or should not be staged\n"
        "- downscore when the idea may be useful but the evidence is too weak for strong confidence\n"
        "- never increase confidence above the reviewer-proposed value\n"
        "- prefer rejecting or downscoring over approving weak evidence\n\n"
        "Candidates to verify:\n"
        f"{json.dumps(candidates, ensure_ascii=False, sort_keys=True)}"
    )



def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    category = str(candidate.get("category", "unknown")).strip().lower()
    if category not in VALID_CATEGORIES:
        category = "unknown"

    summary = " ".join(str(candidate.get("summary", "")).split()).strip()

    try:
        confidence = float(candidate.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    normalized = {
        "category": category,
        "summary": summary,
        "confidence": confidence,
        "reason": str(candidate.get("reason", "")).strip(),
        "target": str(candidate.get("target", "")).strip(),
        "payload": candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {},
    }
    return normalized



def parse_auto_learning_review(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return []

    raw_candidates = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(raw_candidates, list):
        return []

    normalized_candidates = []
    for candidate in raw_candidates:
        if isinstance(candidate, dict):
            normalized_candidates.append(normalize_candidate(candidate))
    return normalized_candidates



def normalize_verifier_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    try:
        index = int(decision.get("index"))
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None

    disposition = str(decision.get("disposition", "reject")).strip().lower()
    if disposition not in VALID_VERIFIER_DISPOSITIONS:
        disposition = "reject"

    try:
        confidence = float(decision.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "index": index,
        "disposition": disposition,
        "confidence": confidence,
        "reason": str(decision.get("reason", "")).strip(),
    }



def parse_auto_learning_verifier_review(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return []

    raw_decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(raw_decisions, list):
        return []

    normalized_decisions = []
    for decision in raw_decisions:
        if not isinstance(decision, dict):
            continue
        normalized = normalize_verifier_decision(decision)
        if normalized is None:
            return []
        normalized_decisions.append(normalized)
    return normalized_decisions



def should_promote_candidate(candidate: dict[str, Any], threshold: float) -> bool:
    try:
        confidence = float(candidate.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return confidence >= threshold
