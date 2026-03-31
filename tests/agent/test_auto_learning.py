import json

from agent.auto_learning import (
    build_auto_learning_review_prompt,
    build_auto_learning_verifier_prompt,
    candidate_semantic_key,
    candidates_semantically_overlap,
    detect_candidate_contradictions,
    normalize_candidate,
    normalize_verifier_decision,
    parse_auto_learning_review,
    parse_auto_learning_verifier_review,
    should_promote_candidate,
)


def test_build_auto_learning_review_prompt_mentions_thresholds_and_supported_hooks():
    prompt = build_auto_learning_review_prompt(
        allow_memory=True,
        allow_skills=False,
        min_tool_iterations=4,
        promotion_threshold=0.8,
    )

    assert "strict JSON" in prompt
    assert '"candidates"' in prompt
    assert '"memory"' in prompt
    assert '"skill"' not in prompt
    assert "4" in prompt
    assert "0.8" in prompt
    assert "failure followed by recovery" in prompt
    assert "explicit user correction" in prompt
    assert "delegated completion" in prompt



def test_build_auto_learning_verifier_prompt_mentions_disposition_contract_and_candidates():
    prompt = build_auto_learning_verifier_prompt(
        candidates=[
            {
                "category": "memory",
                "summary": "User prefers concise responses",
                "confidence": 0.91,
                "reason": "Repeated explicit correction",
                "target": "user",
                "payload": {"action": "add", "content": "User prefers concise responses."},
                "evidence": {"hook_reason": "tool_heavy_success"},
            }
        ],
        promotion_threshold=0.8,
    )

    assert "strict JSON" in prompt
    assert '"decisions"' in prompt
    assert '"approve"' in prompt
    assert '"reject"' in prompt
    assert '"downscore"' in prompt
    assert "User prefers concise responses" in prompt
    assert "0.8" in prompt



def test_normalize_candidate_clamps_confidence_and_validates_category():
    candidate = normalize_candidate(
        {
            "category": "not-real",
            "summary": "  User prefers concise responses  ",
            "confidence": 1.7,
            "reason": "Repeated explicit correction",
            "payload": {"action": "add", "content": "User prefers concise responses."},
        }
    )

    assert candidate["category"] == "unknown"
    assert candidate["summary"] == "User prefers concise responses"
    assert candidate["confidence"] == 1.0
    assert candidate["payload"]["action"] == "add"



def test_parse_auto_learning_review_returns_memory_candidate_from_valid_json():
    text = json.dumps(
        {
            "candidates": [
                {
                    "category": "memory",
                    "summary": "User prefers concise responses",
                    "confidence": 0.93,
                    "reason": "Repeated explicit correction from user",
                    "target": "user",
                    "payload": {"action": "add", "content": "User prefers concise responses."},
                }
            ]
        }
    )

    candidates = parse_auto_learning_review(text)

    assert len(candidates) == 1
    assert candidates[0]["category"] == "memory"
    assert candidates[0]["target"] == "user"
    assert candidates[0]["confidence"] == 0.93



def test_parse_auto_learning_review_returns_skill_candidate_from_valid_json():
    text = json.dumps(
        {
            "candidates": [
                {
                    "category": "skill",
                    "summary": "Patch outdated OpenVINO skill steps",
                    "confidence": 0.88,
                    "reason": "Workflow required iterative fixes",
                    "target": "openvino-qwen-no-think",
                    "payload": {
                        "action": "patch",
                        "old_string": "old step",
                        "new_string": "new step",
                    },
                }
            ]
        }
    )

    candidates = parse_auto_learning_review(text)

    assert len(candidates) == 1
    assert candidates[0]["category"] == "skill"
    assert candidates[0]["payload"]["action"] == "patch"



def test_parse_auto_learning_review_returns_empty_list_on_invalid_json():
    assert parse_auto_learning_review("not valid json") == []



def test_normalize_verifier_decision_clamps_confidence_and_disposition():
    decision = normalize_verifier_decision(
        {
            "index": "2",
            "disposition": "not-real",
            "confidence": 1.7,
            "reason": "  Missing evidence.  ",
        }
    )

    assert decision == {
        "index": 2,
        "disposition": "reject",
        "confidence": 1.0,
        "reason": "Missing evidence.",
    }



def test_parse_auto_learning_verifier_review_returns_normalized_decisions_from_valid_json():
    text = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "disposition": "APPROVE",
                    "confidence": 0.76,
                    "reason": "Evidence supports a durable preference.",
                },
                {
                    "index": 1,
                    "disposition": "downscore",
                    "confidence": 0.41,
                    "reason": "Useful signal, but only one observed instance.",
                },
            ]
        }
    )

    decisions = parse_auto_learning_verifier_review(text)

    assert decisions == [
        {
            "index": 0,
            "disposition": "approve",
            "confidence": 0.76,
            "reason": "Evidence supports a durable preference.",
        },
        {
            "index": 1,
            "disposition": "downscore",
            "confidence": 0.41,
            "reason": "Useful signal, but only one observed instance.",
        },
    ]



def test_parse_auto_learning_verifier_review_returns_empty_list_on_invalid_or_missing_decisions():
    assert parse_auto_learning_verifier_review("not valid json") == []
    assert parse_auto_learning_verifier_review(json.dumps({"decisions": "nope"})) == []
    assert parse_auto_learning_verifier_review(json.dumps({"decisions": [{"index": "bad"}]})) == []



def test_should_promote_candidate_uses_threshold():
    candidate = normalize_candidate(
        {
            "category": "memory",
            "summary": "User prefers concise responses",
            "confidence": 0.81,
            "payload": {"action": "add", "content": "User prefers concise responses."},
        }
    )

    assert should_promote_candidate(candidate, 0.8) is True
    assert should_promote_candidate(candidate, 0.9) is False



def test_candidate_semantic_key_normalizes_wording_variants():
    first = candidate_semantic_key(
        {
            "category": "memory",
            "summary": "User prefers concise responses",
            "target": "user",
            "payload": {"action": "add", "content": "User prefers concise responses."},
        }
    )
    second = candidate_semantic_key(
        {
            "category": "memory",
            "summary": "User likes brief answers",
            "target": "user",
            "payload": {"action": "add", "content": "User likes brief answers."},
        }
    )

    assert first == second
    assert "memory" in first
    assert "user" in first



def test_candidates_semantically_overlap_on_wording_variants():
    assert candidates_semantically_overlap(
        {
            "category": "memory",
            "summary": "User prefers concise responses",
            "target": "user",
            "payload": {"action": "add", "content": "User prefers concise responses."},
        },
        {
            "category": "memory",
            "summary": "User likes brief answers",
            "target": "user",
            "payload": {"action": "add", "content": "User likes brief answers."},
        },
    ) is True



def test_detect_candidate_contradictions_against_durable_entries():
    contradiction = detect_candidate_contradictions(
        {
            "category": "memory",
            "summary": "User prefers verbose responses",
            "target": "user",
            "payload": {"action": "add", "content": "User prefers verbose responses."},
        },
        durable_entries=["User prefers concise responses"],
        staged_candidates=[
            {
                "category": "memory",
                "summary": "User prefers brief answers",
                "target": "user",
            }
        ],
    )

    assert contradiction["has_contradiction"] is True
    assert contradiction["review_required"] is True
    assert contradiction["matches"]
