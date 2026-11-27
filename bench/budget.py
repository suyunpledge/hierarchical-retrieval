"""A shared, transactional USD cap for all paid requests in one experiment."""

import json
import sqlite3
import uuid


class Budget:
    def __init__(self, path, cap):
        self.path = path
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS spends (id TEXT PRIMARY KEY, units INTEGER NOT NULL, receipt TEXT UNIQUE)"
            )
            db.execute("INSERT OR IGNORE INTO account VALUES (1,?)", (round(cap * 1e7),))
            if db.execute("SELECT cap FROM account").fetchone()[0] != round(cap * 1e7):
                raise ValueError("Budget does not match existing ledger")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def pin(self, label, digest):
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS inputs(label TEXT PRIMARY KEY,digest TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO inputs VALUES(?,?)", (label, digest))
            if db.execute("SELECT digest FROM inputs WHERE label=?", (label,)).fetchone()[0] != digest:
                raise ValueError("Experiment inputs changed; use the matching snapshot or a new directory")

    def reserve_units(self, units):
        if not isinstance(units, int) or units <= 0:
            raise ValueError("A request needs a positive maximum charge")
        key = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            room = db.execute(
                "SELECT cap-(SELECT coalesce(sum(units),0) FROM spends) FROM account"
            ).fetchone()[0]
            if units > room:
                raise RuntimeError("Experiment budget reached; no request sent")
            db.execute("INSERT INTO spends VALUES(?,?,NULL)", (key, units))
        return key

    def reserve(self, body):
        # UTF-8 byte count bounds text tokens. Allow protocol framing and charge
        # every possible input token at the highest enabled (30-minute write) rate.
        size = len(json.dumps(body, ensure_ascii=False).encode()) + 256 + 32 * len(body["messages"])
        return self.reserve_units(25 * size + 100 * (body.get("max_completion_tokens") or body["max_tokens"]))

    def record(self, key, units, receipt):
        with self.connect() as db:
            old = db.execute("SELECT units FROM spends WHERE id=? AND receipt IS NULL", (key,)).fetchone()
            if old is None or not isinstance(units, int) or not 0 <= units <= old[0]:
                raise RuntimeError("Invalid or exceeded request reservation")
            db.execute("UPDATE spends SET units=?,receipt=? WHERE id=?", (units, receipt, key))

    def settle(self, key, response):
        if response.ok:
            data = response.json()
            u = data["usage"]
            d = u["prompt_tokens_details"]
            r = d["cached_tokens"]
            w = d["cache_write_tokens"]
            self.record(
                key, 20 * (u["prompt_tokens"] - r - w) + r + 25 * w + 100 * u["completion_tokens"], data["id"]
            )
        elif response.status_code in (400, 401, 403, 404, 429):
            self.record(key, 0, "rejected-" + key)
        # An unconfirmed delivery keeps its entire reservation until reconciled.
