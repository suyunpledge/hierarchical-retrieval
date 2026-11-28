import json
from contextlib import closing

import api_server
import pytest
from fastapi.testclient import TestClient
from hierarchical_retrieval.core.tool_receipts import ReceiptScope, ToolReceipt, ToolReceiptStore

FOCUS = ReceiptScope(run_id="work", resource="agenda", field="selection")
TARGET = {"resource": "blue", "field": "route"}


def bind(store, version, targets):
    store.record(
        ToolReceipt(
            **FOCUS.model_dump(),
            revision=version,
            receipt_id=f"d{version}",
            status="committed",
            value=json.dumps({"targets": targets}),
        )
    )


def event(store, rev, value, *, run="work", status="committed", source=None):
    store.record(
        ToolReceipt(
            **TARGET, run_id=run, revision=rev, receipt_id=source or str(rev), status=status, value=value
        )
    )


def test_exact_memory_survives_handoff_and_revisions(tmp_path):
    store = ToolReceiptStore(tmp_path)
    bind(store, 4, [{**TARGET, "revision": 0}])
    bind(store, 2, [TARGET])
    event(store, 3, "new")
    event(store, 0, "other", run="neighbor")
    assert store.resolve_task_context(FOCUS)["status"] == "missing"
    event(store, 0, "first")
    reopened = ToolReceiptStore(tmp_path)
    result = reopened.resolve_task_context(FOCUS)
    assert result["decision"]["receipt_id"] == "d4" and result["evidence"]["value"] == "first"
    event(store, 0, "different", source="other-event")
    assert reopened.resolve_task_context(FOCUS)["status"] == "conflict"
    bind(store, 5, [TARGET])
    assert reopened.resolve_task_context(FOCUS)["evidence"]["value"] == "new"
    event(store, 4, None, status="unknown")
    assert reopened.resolve_task_context(FOCUS)["status"] == "uncertain"


@pytest.mark.parametrize(
    ("targets", "status"),
    [
        ([], "cleared"),
        ([TARGET, {**TARGET, "revision": 0}], "clarify"),
        ([{**TARGET, "run_id": "different"}], "invalid_selection"),
        ([{**TARGET, "revision": True}], "invalid_selection"),
    ],
)
def test_unresolved_context_exposes_no_actionable_values(tmp_path, targets, status):
    store = ToolReceiptStore(tmp_path)
    event(store, 1, "do-not-use")
    bind(store, 1, targets)
    result = store.resolve_task_context(FOCUS)
    assert result["status"] == status and "evidence" not in result
    assert "do-not-use" not in json.dumps(result)


def test_same_read_snapshot_for_decision_and_evidence(tmp_path, monkeypatch):
    store = ToolReceiptStore(tmp_path)
    with closing(store.connect()) as db:
        db.execute("PRAGMA journal_mode=WAL").fetchone()
    bind(store, 1, [TARGET, TARGET])
    event(store, 1, "original")
    lookup = store._lookup

    def interleave(db, scope, version=None):
        result = lookup(db, scope, version)
        if scope == FOCUS:
            bind(store, 2, [])
            event(store, 2, "updated")
        return result

    monkeypatch.setattr(store, "_lookup", interleave)
    result = store.resolve_task_context(FOCUS)
    assert result["decision"]["revision"] == 1 and result["evidence"]["value"] == "original"
    assert ToolReceiptStore(tmp_path).resolve_task_context(FOCUS)["status"] == "cleared"


def test_resolver_uses_service_auth_and_reports_absent_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server, "API_TOKEN", "for-test")
    with TestClient(api_server.app) as client:
        url = "/task-context/resolve"
        assert client.post(url, json=FOCUS.model_dump()).status_code == 401
        r = client.post(url, headers={"Authorization": "Bearer for-test"}, json=FOCUS.model_dump())
        assert r.status_code == 200 and r.json()["decision"]["status"] == "missing"
