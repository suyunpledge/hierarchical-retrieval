"""Plain paragraph retrieval using the existing local BGE-M3 embedding service."""

import hashlib
import re
import sqlite3

import numpy as np
import requests


class ParagraphSearch:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key BLOB PRIMARY KEY, value BLOB NOT NULL)")
        self.http = requests.Session()
        url = "http://127.0.0.1:8101/v1"
        models = self.http.get(url + "/models", timeout=10).json()["data"]
        self.model = next(m["id"] for m in models if "bge-m3" in m["id"].lower())
        self.url = url + "/embeddings"

    def embed(self, texts):
        keys = [hashlib.sha256((self.model + "\n" + s).encode()).digest() for s in texts]
        values = {}
        for key in set(keys):
            row = self.db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
            if row:
                values[key] = np.frombuffer(row[0], dtype="float32")
        missing = {k: s for k, s in zip(keys, texts) if k not in values}
        items = list(missing)
        for start in range(0, len(items), 12):
            batch = items[start : start + 12]
            r = self.http.post(
                self.url, json={"model": self.model, "input": [missing[k] for k in batch]}, timeout=90
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda item: item["index"])
            if len(data) != len(batch):
                raise ValueError("Incomplete embedding batch")
            with self.db:
                for key, entry in zip(batch, data):
                    v = np.array(entry["embedding"], dtype="float32")
                    norm = np.linalg.norm(v)
                    if v.shape != (1024,) or not np.isfinite(norm) or norm == 0:
                        raise ValueError("Invalid embedding")
                    v /= norm
                    values[key] = v
                    self.db.execute("INSERT OR IGNORE INTO vectors VALUES (?,?)", (key, v.tobytes()))
        return np.stack([values[k] for k in keys])

    def __call__(self, system, docs, ids, events, chapter, question, header):
        paragraphs = []
        recent = []
        for _, _, action, content, answer in events:
            recent.append({"role": "user", "content": content})
            if answer is not None and action != "note":
                recent.append({"role": "assistant", "content": answer})
            if action == "note":
                paragraphs.append(content)
            if action == "read":
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                title = next((line for line in lines if re.match(r"第\s*\d+\s*章", line)), "")
                paragraphs.extend(title + "\n" + line for line in lines[2:])
        paragraphs = list(dict.fromkeys(paragraphs))
        vectors = self.embed([question, *paragraphs])
        top = np.argsort(-(vectors[1:] @ vectors[0]), kind="stable")[:4]
        packed = "检索材料\n" + "\n\n".join(paragraphs[i] for i in top)
        return [
            header,
            {"role": "user", "content": packed},
            *recent[-6:],
            {"role": "user", "content": question},
        ]
