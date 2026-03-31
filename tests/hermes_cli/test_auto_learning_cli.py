import sys

from tools.auto_learning_store import AutoLearningStore



def test_cli_autolearning_enable_routes_to_handler(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_auto_learning(args):
        captured["action"] = args.auto_learning_action

    monkeypatch.setattr(main_mod, "cmd_auto_learning", fake_cmd_auto_learning)
    monkeypatch.setattr(sys, "argv", ["hermes", "autolearning", "enable"])

    main_mod.main()

    assert captured == {"action": "enable"}



def test_cli_autolearning_list_routes_status_filter(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_auto_learning(args):
        captured["action"] = args.auto_learning_action
        captured["status"] = args.status

    monkeypatch.setattr(main_mod, "cmd_auto_learning", fake_cmd_auto_learning)
    monkeypatch.setattr(sys, "argv", ["hermes", "autolearning", "list", "--status", "candidate"])

    main_mod.main()

    assert captured == {"action": "list", "status": "candidate"}



def test_cli_autolearning_search_routes_query(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_auto_learning(args):
        captured["action"] = args.auto_learning_action
        captured["query"] = args.query

    monkeypatch.setattr(main_mod, "cmd_auto_learning", fake_cmd_auto_learning)
    monkeypatch.setattr(sys, "argv", ["hermes", "autolearning", "search", "concise"])

    main_mod.main()

    assert captured == {"action": "search", "query": "concise"}



def test_auto_learning_command_enable_updates_config(monkeypatch):
    from hermes_cli.auto_learning import auto_learning_command

    calls = []

    monkeypatch.setattr("hermes_cli.auto_learning.set_config_value", lambda key, value: calls.append((key, value)))

    class Args:
        auto_learning_action = "enable"

    auto_learning_command(Args())

    assert calls == [("auto_learning.enabled", "true")]



def test_auto_learning_command_disable_updates_config(monkeypatch):
    from hermes_cli.auto_learning import auto_learning_command

    calls = []

    monkeypatch.setattr("hermes_cli.auto_learning.set_config_value", lambda key, value: calls.append((key, value)))

    class Args:
        auto_learning_action = "disable"

    auto_learning_command(Args())

    assert calls == [("auto_learning.enabled", "false")]



def test_auto_learning_status_surfaces_reviewer_and_verifier_routing(tmp_path, monkeypatch, capsys):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr(
        "hermes_cli.auto_learning.load_config",
        lambda: {
            "model": "anthropic/claude-opus-4.6",
            "auto_learning": {
                "enabled": True,
                "reviewer": {
                    "provider": "openrouter",
                    "model": "google/gemini-3-flash-preview",
                    "api_key": "sk-secret-reviewer",
                    "max_iterations": 6,
                    "timeout": 45,
                },
                "verifier": {},
            },
        },
    )

    class Args:
        auto_learning_action = "status"

    auto_learning_command(Args())

    out = capsys.readouterr().out
    assert "Karpathy auto-learning: enabled" in out
    assert "Main model: anthropic/claude-opus-4.6" in out
    assert "Reviewer: provider=openrouter; model=google/gemini-3-flash-preview; max_iterations=6; timeout=45s" in out
    assert "Verifier: inherit main agent route" in out
    assert "sk-secret-reviewer" not in out





def test_auto_learning_status_surfaces_hook_and_verifier_breakdowns(tmp_path, monkeypatch, capsys):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    approved = store.add_candidate(
        category="memory",
        summary="User prefers concise responses",
        confidence=0.84,
        evidence={
            "hook_reason": "explicit_user_correction",
            "verifier": {"disposition": "approve", "confidence": 0.84, "reason": "strong evidence"},
        },
    )
    store.mark_status(approved["id"], "promoted")
    rejected = store.add_candidate(
        category="skill",
        summary="Patch retry workflow",
        confidence=0.31,
        evidence={
            "hook_reason": "failure_recovery",
            "verifier": {"disposition": "reject", "confidence": 0.0, "reason": "too weak"},
        },
    )
    store.mark_status(rejected["id"], "rejected")

    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr(
        "hermes_cli.auto_learning.load_config",
        lambda: {
            "model": "anthropic/claude-opus-4.6",
            "auto_learning": {"enabled": True, "reviewer": {}, "verifier": {}},
        },
    )

    class Args:
        auto_learning_action = "status"

    auto_learning_command(Args())

    out = capsys.readouterr().out
    assert "Hook reasons:" in out
    assert "explicit_user_correction: 1" in out
    assert "failure_recovery: 1" in out
    assert "Verifier outcomes:" in out
    assert "approve: 1" in out
    assert "reject: 1" in out



def test_auto_learning_list_surfaces_hook_and_verifier_details(tmp_path, monkeypatch, capsys):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers concise responses",
        confidence=0.41,
        evidence={
            "hook_reason": "failure_recovery",
            "verifier": {"disposition": "downscore", "confidence": 0.41, "reason": "single recovered case"},
        },
    )

    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr("hermes_cli.auto_learning.load_config", lambda: {"auto_learning": {"enabled": True}})

    class Args:
        auto_learning_action = "list"
        status = None

    auto_learning_command(Args())

    out = capsys.readouterr().out
    assert f"{entry['id']}  [candidate]  memory  User prefers concise responses" in out
    assert "hook=failure_recovery" in out
    assert "verifier=downscore@0.41" in out



def test_auto_learning_command_promote_and_reject_update_store(tmp_path, monkeypatch):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers concise responses",
        confidence=0.8,
        evidence={"source": "test"},
    )

    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr("hermes_cli.auto_learning.load_config", lambda: {"auto_learning": {"enabled": True}})

    class PromoteArgs:
        auto_learning_action = "promote"
        id = entry["id"]

    auto_learning_command(PromoteArgs())
    assert store.list_candidates(status="promoted")[0]["id"] == entry["id"]

    second = store.add_candidate(
        category="memory",
        summary="Another candidate",
        confidence=0.7,
        evidence={"source": "test"},
    )

    class RejectArgs:
        auto_learning_action = "reject"
        id = second["id"]

    auto_learning_command(RejectArgs())
    assert store.list_candidates(status="rejected")[0]["id"] == second["id"]





def test_auto_learning_status_surfaces_manual_review_and_shadow_breakdowns(tmp_path, monkeypatch, capsys):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers verbose responses",
        confidence=0.52,
        evidence={
            "quality": {
                "shadow_decision": "manual_review",
                "review_required": True,
                "contradictions": {"has_contradiction": True},
            }
        },
        target="user",
    )
    store.mark_status(entry["id"], "manual_review")

    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr(
        "hermes_cli.auto_learning.load_config",
        lambda: {
            "model": "anthropic/claude-opus-4.6",
            "auto_learning": {"enabled": True, "reviewer": {}, "verifier": {}},
        },
    )

    class Args:
        auto_learning_action = "status"

    auto_learning_command(Args())

    out = capsys.readouterr().out
    assert "manual_review: 1" in out
    assert "Quality outcomes:" in out
    assert "manual_review: 1" in out



def test_auto_learning_search_and_show_surface_candidate_details(tmp_path, monkeypatch, capsys):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers concise responses",
        confidence=0.73,
        evidence={"quality": {"semantic_key": "memory|user|concise|response"}, "hook_reason": "explicit_user_correction"},
        target="user",
    )

    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr("hermes_cli.auto_learning.load_config", lambda: {"auto_learning": {"enabled": True}})

    class SearchArgs:
        auto_learning_action = "search"
        query = "concise"
        status = None

    auto_learning_command(SearchArgs())
    search_out = capsys.readouterr().out
    assert entry["id"] in search_out

    class ShowArgs:
        auto_learning_action = "show"
        id = entry["id"]

    auto_learning_command(ShowArgs())
    show_out = capsys.readouterr().out
    assert "semantic_key" in show_out
    assert "explicit_user_correction" in show_out
