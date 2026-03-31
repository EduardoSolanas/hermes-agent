from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.auto_learning import candidate_semantic_key, candidates_semantically_overlap
from hermes_cli.config import get_hermes_home


VALID_STATUSES = {"candidate", "promoted", "rejected", "superseded", "manual_review"}
VALID_CATEGORIES = {"memory", "skill", "unknown"}


class AutoLearningStore:
    def __init__(self, path: Path | None = None, max_entries: int = 200):
        self.path = Path(path) if path is not None else get_hermes_home() / "auto_learning" / "candidates.jsonl"
        self.max_entries = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add_candidate(
        self,
        *,
        category: str,
        summary: str,
        confidence: float,
        evidence: dict,
        action: str | None = None,
        target: str | None = None,
        payload: dict | None = None,
    ) -> dict[str, Any]:
        normalized_category = category if category in VALID_CATEGORIES else "unknown"
        normalized_summary = summary.strip()
        normalized_payload = payload if isinstance(payload, dict) else {}
        fingerprint = self._make_fingerprint(
            category=normalized_category,
            summary=normalized_summary,
            action=action,
            target=target,
        )

        existing = self.find_by_fingerprint(fingerprint)
        if existing is not None:
            return existing

        semantic_key = candidate_semantic_key(
            {
                "category": normalized_category,
                "summary": normalized_summary,
                "target": target or "",
                "payload": normalized_payload,
            }
        )

        entry = {
            "id": f"al-{uuid4().hex[:12]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "candidate",
            "category": normalized_category,
            "summary": normalized_summary,
            "confidence": confidence,
            "evidence": evidence,
            "fingerprint": fingerprint,
            "semantic_key": semantic_key,
        }
        if action is not None:
            entry["action"] = action
        if target is not None:
            entry["target"] = target
        if payload is not None:
            entry["payload"] = payload

        items = self._load_items()
        overlap = self._find_semantic_overlap(items, entry)
        if overlap is not None:
            if float(entry.get("confidence", 0.0) or 0.0) > float(overlap.get("confidence", 0.0) or 0.0):
                overlap["status"] = "superseded"
                overlap["superseded_by"] = entry["id"]
                entry["supersedes"] = overlap["id"]
            else:
                entry["status"] = "superseded"
                entry["superseded_by"] = overlap["id"]
                overlap.setdefault("supersedes", [])
        items.append(entry)
        self._save_items(items)
        return entry

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        items = self._load_items()
        if status is None:
            return items
        return [item for item in items if item.get("status") == status]

    def get_candidate(self, entry_id: str) -> dict[str, Any] | None:
        for item in self._load_items():
            if item.get("id") == entry_id:
                return item
        return None

    def mark_status(self, entry_id: str, status: str, note: str | None = None) -> dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        items = self._load_items()
        for item in items:
            if item.get("id") == entry_id:
                item["status"] = status
                if note is not None:
                    item["promotion_note"] = note
                self._save_items(items)
                return item
        raise KeyError(entry_id)

    def update_candidate(self, entry_id: str, **updates: Any) -> dict[str, Any]:
        items = self._load_items()
        for item in items:
            if item.get("id") == entry_id:
                item.update(updates)
                self._save_items(items)
                return item
        raise KeyError(entry_id)

    def find_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        for item in self._load_items():
            if item.get("fingerprint") == fingerprint:
                return item
        return None

    def search_candidates(self, query: str, status: str | None = None) -> list[dict[str, Any]]:
        tokens = [token for token in str(query or "").strip().lower().split() if token]
        if not tokens:
            return self.list_candidates(status=status)

        matches: list[dict[str, Any]] = []
        for item in self.list_candidates(status=status):
            haystack = json.dumps(item, sort_keys=True).lower()
            if all(token in haystack for token in tokens):
                matches.append(item)
        return matches

    def _find_semantic_overlap(self, items: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any] | None:
        for item in reversed(items):
            if item.get("status") not in {"candidate", "manual_review", "promoted"}:
                continue
            if item.get("category") != entry.get("category"):
                continue
            if candidates_semantically_overlap(item, entry):
                return item
        return None

    def _load_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            items.append(json.loads(line))
        return items

    def _save_items(self, items: list[dict[str, Any]]) -> None:
        trimmed = items[-self.max_entries :]
        with self.path.open("w", encoding="utf-8") as handle:
            for item in trimmed:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    @staticmethod
    def _make_fingerprint(*, category: str, summary: str, action: str | None, target: str | None) -> str:
        normalized = {
            "category": category,
            "summary": " ".join(summary.split()).strip().lower(),
            "action": (action or "").strip().lower(),
            "target": (target or "").strip().lower(),
        }
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
