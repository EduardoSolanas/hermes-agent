from __future__ import annotations

from pathlib import Path
import json

from hermes_cli.config import load_config, set_config_value
from tools.auto_learning_store import AutoLearningStore


LABEL = "Karpathy auto-learning"


def _load_store() -> AutoLearningStore:
    config = load_config()
    auto_cfg = config.get("auto_learning", {}) or {}
    store_path = auto_cfg.get("store_path") or None
    max_entries = int(auto_cfg.get("candidate_max_entries", 200) or 200)
    return AutoLearningStore(path=Path(store_path) if store_path else None, max_entries=max_entries)


def _format_actor_route(label: str, actor_cfg: dict | None) -> list[str]:
    actor_cfg = actor_cfg if isinstance(actor_cfg, dict) else {}
    model = str(actor_cfg.get("model") or "").strip()
    provider = str(actor_cfg.get("provider") or "").strip()
    base_url = str(actor_cfg.get("base_url") or "").strip()
    max_iterations = actor_cfg.get("max_iterations")
    timeout = actor_cfg.get("timeout")

    details: list[str] = []
    if base_url:
        details.append(f"endpoint={base_url}")
    elif provider:
        details.append(f"provider={provider}")
    if model:
        details.append(f"model={model}")
    if max_iterations not in (None, "", 0, "0"):
        details.append(f"max_iterations={max_iterations}")
    if timeout not in (None, "", 0, "0"):
        details.append(f"timeout={timeout}s")

    if not details:
        return [f"{label}: inherit main agent route"]
    return [f"{label}: {'; '.join(details)}"]


def _count_by_status(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("status") or "unknown").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_hook_reason(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        key = str(evidence.get("hook_reason") or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_verifier_outcome(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        verifier = evidence.get("verifier") if isinstance(evidence.get("verifier"), dict) else {}
        key = str(verifier.get("disposition") or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_quality_outcome(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        quality = evidence.get("quality") if isinstance(evidence.get("quality"), dict) else {}
        key = str(quality.get("shadow_decision") or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _print_count_block(label: str, counts: dict[str, int]) -> None:
    if not counts:
        return
    print(label)
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")


def _format_verifier_summary(verifier: dict | None) -> str | None:
    verifier = verifier if isinstance(verifier, dict) else {}
    disposition = str(verifier.get("disposition") or "").strip()
    if not disposition:
        return None
    try:
        confidence = float(verifier.get("confidence"))
    except (TypeError, ValueError):
        return disposition
    return f"{disposition}@{confidence:.2f}"


def _format_candidate_inspection(item: dict) -> str | None:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    details: list[str] = []

    hook_reason = str(evidence.get("hook_reason") or "").strip()
    if hook_reason:
        details.append(f"hook={hook_reason}")

    verifier_summary = _format_verifier_summary(evidence.get("verifier"))
    if verifier_summary:
        details.append(f"verifier={verifier_summary}")

    quality = evidence.get("quality") if isinstance(evidence.get("quality"), dict) else {}
    semantic_key = str(quality.get("semantic_key") or "").strip()
    if semantic_key:
        details.append(f"semantic_key={semantic_key}")

    skill_validation = quality.get("skill_validation") if isinstance(quality.get("skill_validation"), dict) else {}
    if skill_validation:
        action = str(skill_validation.get("action") or "unknown").strip() or "unknown"
        name = str(skill_validation.get("name") or "").strip()
        validity = "valid" if skill_validation.get("valid") else "invalid"
        label = f"skill_validation={action}:{validity}"
        if name:
            label += f"@{name}"
        details.append(label)
        error = str(skill_validation.get("error") or "").strip()
        if error:
            details.append(error)

    if not details:
        return None
    return "  " + "  ".join(details)


def auto_learning_command(args) -> None:
    action = getattr(args, "auto_learning_action", None) or "status"
    config = load_config()
    auto_cfg = config.get("auto_learning", {}) or {}

    if action == "status":
        store = _load_store()
        items = store.list_candidates()
        enabled = bool(auto_cfg.get("enabled", False))
        print(f"{LABEL}: {'enabled' if enabled else 'disabled'}")
        print(f"Store: {store.path}")
        print(f"Main model: {config.get('model', '(unset)')}")
        for line in _format_actor_route("Reviewer", auto_cfg.get("reviewer")):
            print(line)
        for line in _format_actor_route("Verifier", auto_cfg.get("verifier")):
            print(line)
        print(f"Candidates: {len(items)}")
        _print_count_block("Statuses:", _count_by_status(items))
        _print_count_block("Hook reasons:", _count_by_hook_reason(items))
        _print_count_block("Verifier outcomes:", _count_by_verifier_outcome(items))
        _print_count_block("Quality outcomes:", _count_by_quality_outcome(items))
        return

    if action == "enable":
        set_config_value("auto_learning.enabled", "true")
        print(f"{LABEL} enabled.")
        return

    if action == "disable":
        set_config_value("auto_learning.enabled", "false")
        print(f"{LABEL} disabled.")
        return

    store = _load_store()

    if action == "list":
        items = store.list_candidates(status=getattr(args, "status", None))
        if not items:
            print("No staged auto-learning candidates.")
            return
        for item in items:
            print(f"{item['id']}  [{item.get('status', 'candidate')}]  {item.get('category', 'unknown')}  {item.get('summary', '')}")
            inspection = _format_candidate_inspection(item)
            if inspection:
                print(inspection)
        return

    if action == "search":
        items = store.search_candidates(getattr(args, "query", ""), status=getattr(args, "status", None))
        if not items:
            print("No staged auto-learning candidates matched.")
            return
        for item in items:
            print(f"{item['id']}  [{item.get('status', 'candidate')}]  {item.get('category', 'unknown')}  {item.get('summary', '')}")
            inspection = _format_candidate_inspection(item)
            if inspection:
                print(inspection)
        return

    if action == "show":
        item = store.get_candidate(args.id)
        if not item:
            raise SystemExit(f"Unknown candidate id: {args.id}")
        print(json.dumps(item, indent=2, ensure_ascii=False, sort_keys=True))
        return

    if action == "promote":
        updated = store.mark_status(args.id, "promoted", note="manual CLI promotion")
        print(f"Promoted {updated['id']}: {updated.get('summary', '')}")
        return

    if action == "reject":
        updated = store.mark_status(args.id, "rejected", note="manual CLI rejection")
        print(f"Rejected {updated['id']}: {updated.get('summary', '')}")
        return

    raise SystemExit(f"Unknown auto-learning action: {action}")
