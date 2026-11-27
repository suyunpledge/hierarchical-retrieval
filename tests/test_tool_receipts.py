from concurrent.futures import ThreadPoolExecutor

import api_server
import pytest
from fastapi.testclient import TestClient
from hierarchical_retrieval.core.tool_receipts import (
    ReceiptIdentityConflict,
    ReceiptScope,
    ToolReceipt,
    ToolReceiptStore,
)


def receipt(revision, source, value=None, status="committed", **scope):
    subject = {"run_id": "batch-A", "resource": "blue", "field": "route", **scope}
    return ToolReceipt(**subject, revision=revision, receipt_id=source, value=value, status=status)


def test_concurrent_replay_survives_reopening_and_scopes_do_not_collide(tmp_path):
    store = ToolReceiptStore(tmp_path)
    older = receipt(1, "old", "r1")
    newest = receipt(2, "new", "r2")
    attempts = [
        newest,
        older,
        receipt(8, "new", "other", run_id="batch-B"),
        receipt(8, "new", "elsewhere", resource="red"),
        receipt(8, "new", "reviewer", field="reviewer"),
        receipt(7, "bad", "proposed", status="not_started"),
    ] * 2
    with ThreadPoolExecutor(max_workers=4) as pool:
        written = list(pool.map(store.record, attempts))
    assert sum(r["inserted"] for r in written) == 6
    scope = ReceiptScope(run_id="batch-A", resource="blue", field="route")
    assert ToolReceiptStore(tmp_path).lookup(scope) == {
        **scope.model_dump(),
        "status": "committed",
        "revision": 2,
        "receipt_id": "new",
        "value": "r2",
    }
    with pytest.raises(ReceiptIdentityConflict):
        store.record(receipt(2, "new", "changed"))
    assert store.lookup(scope)["value"] == "r2"


def test_http_lookup_withholds_parameters_until_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.config, "storage_root", str(tmp_path))
    monkeypatch.setattr(api_server, "API_TOKEN", "test-receipts")
    with TestClient(api_server.app) as http:
        scope = {"run_id": "batch-A", "resource": "blue", "field": "route"}
        assert http.post("/tool-receipts/lookup", json=scope).status_code == 401
        http.headers["Authorization"] = "Bearer test-receipts"

        def put(row):
            return http.post("/tool-receipts", json=row.model_dump()).status_code

        def get():
            return http.post("/tool-receipts/lookup", json=scope).json()

        assert get() == {**scope, "status": "missing"}
        assert put(receipt(2, "x", "v")) == 200
        assert put(receipt(2, "x", "another")) == 409
        assert put(receipt(2, "y", "different")) == 200
        assert get() == {**scope, "status": "conflict", "revision": 2}
        assert put(receipt(3, "z", status="unknown")) == 200
        assert get() == {**scope, "status": "uncertain", "revision": 3}
        assert put(receipt(4, "fixed", "")) == 200
        assert get()["value"] == ""
        assert (
            http.post(
                "/tool-receipts", json={**receipt(5, "bad", "x").model_dump(), "revision": True}
            ).status_code
            == 422
        )
