"""Durable exact evidence for action parameters returned by tools."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ResourceField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    resource: str = Field(min_length=1, max_length=512)
    field: str = Field(min_length=1, max_length=128)


class ReceiptScope(ResourceField):
    run_id: str = Field(min_length=1, max_length=256)


class FocusTarget(ResourceField):
    revision: int | None = Field(default=None, ge=0, le=2**63 - 1)


class TaskSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    targets: list[FocusTarget] = Field(max_length=32)


class ToolReceipt(ReceiptScope):
    revision: int = Field(ge=0, le=2**63 - 1)
    receipt_id: str = Field(min_length=1, max_length=256)
    status: Literal["committed", "not_started", "unknown"]
    value: str | None = Field(default=None, max_length=8192)

    @model_validator(mode="after")
    def require_committed_value(self):
        if self.status == "committed" and self.value is None:
            raise ValueError("a committed receipt requires a value")
        return self


class ReceiptIdentityConflict(ValueError):
    """The same source receipt was delivered with different contents."""


class ToolReceiptStore:
    def __init__(self, root):
        folder = Path(root)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "tool_receipts.sqlite3"
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS receipts (
                run_id TEXT NOT NULL, resource TEXT NOT NULL, field TEXT NOT NULL,
                receipt_id TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL,
                value TEXT, PRIMARY KEY(run_id, resource, field, receipt_id))""")
            db.execute("""CREATE INDEX IF NOT EXISTS receipt_versions
                ON receipts(run_id, resource, field, revision DESC)
                WHERE status != 'not_started'""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record(self, r: ToolReceipt):
        key = (r.run_id, r.resource, r.field, r.receipt_id)
        body = (r.revision, r.status, r.value)
        with closing(self.connect()) as db, db:
            inserted = db.execute(
                """INSERT INTO receipts
                (run_id,resource,field,receipt_id,revision,status,value) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(run_id,resource,field,receipt_id) DO NOTHING""",
                key + body,
            )
            if not inserted.rowcount:
                previous = db.execute(
                    """SELECT revision,status,value FROM receipts
                    WHERE run_id=? AND resource=? AND field=? AND receipt_id=?""",
                    key,
                ).fetchone()
                if previous != body:
                    raise ReceiptIdentityConflict("receipt_id already recorded with different contents")
        return {"receipt_id": r.receipt_id, "inserted": bool(inserted.rowcount)}

    def lookup(self, scope: ReceiptScope):
        with closing(self.connect()) as db:
            return self._lookup(db, scope)

    @staticmethod
    def _lookup(db, scope, exact_revision=None):
        key = (scope.run_id, scope.resource, scope.field)
        revision, uncertain, first, last, source = db.execute(
            """
            SELECT max(revision), max(status='unknown'), min(value), max(value), min(receipt_id)
            FROM receipts WHERE run_id=? AND resource=? AND field=? AND status!='not_started'
            AND revision=coalesce(?, (SELECT max(revision) FROM receipts WHERE
                run_id=? AND resource=? AND field=? AND status!='not_started'))
            """,
            (*key, exact_revision, *key),
        ).fetchone()
        result = scope.model_dump()
        if revision is None:
            return {**result, "status": "missing"}
        result["revision"] = revision
        if uncertain:
            return {**result, "status": "uncertain"}
        if first != last:
            return {**result, "status": "conflict"}
        return {**result, "status": "committed", "value": first, "receipt_id": source}

    def resolve_task_context(self, decision_scope: ReceiptScope):
        """Resolve a recorded user selection and its evidence from a single read snapshot."""
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            decision = self._lookup(db, decision_scope)
            if decision["status"] != "committed":
                return {"status": decision["status"], "decision": decision}
            value = decision.pop("value")
            try:
                selection = TaskSelection.model_validate_json(value)
            except ValidationError:
                return {"status": "invalid_selection", "decision": decision}

            # Repeated references to the same target do not create an ambiguity.
            candidates = list(dict.fromkeys((t.resource, t.field, t.revision) for t in selection.targets))
            if not candidates:
                return {"status": "cleared", "decision": decision}
            if len(candidates) > 1:
                return {
                    "status": "clarify",
                    "decision": decision,
                    "candidates": [{"resource": r, "field": f, "revision": v} for r, f, v in candidates],
                }

            resource, field, version = candidates[0]
            scope = ReceiptScope(run_id=decision_scope.run_id, resource=resource, field=field)
            evidence = self._lookup(db, scope, version)
            return {
                "status": evidence["status"],
                "decision": decision,
                "target": {"resource": resource, "field": field, "revision": version},
                "evidence": evidence,
            }
