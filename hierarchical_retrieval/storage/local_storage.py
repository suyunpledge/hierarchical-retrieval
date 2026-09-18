"""
本地文件存储实现
"""

import json
import os
import pickle
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from .base import StorageBase


class LocalStorage(StorageBase):
    """基于本地文件系统的键值存储"""

    def __init__(self, root_dir: str, use_json: bool = True):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.use_json = use_json
        self._lock = threading.RLock()

    def _resolve_path(self, key: str) -> Path:
        # 安全地将 key 转为文件路径
        safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        ext = ".json" if self.use_json else ".pkl"
        return self.root / f"{safe}{ext}"

    def save(self, key: str, value: Any) -> None:
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with self._lock:
            try:
                if self.use_json:
                    tmp.write_text(
                        json.dumps(value, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
                else:
                    tmp.write_bytes(pickle.dumps(value))
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()

    def load(self, key: str) -> Optional[Any]:
        path = self._resolve_path(key)
        if not path.exists():
            return None
        if self.use_json:
            return json.loads(path.read_text(encoding="utf-8"))
        return pickle.loads(path.read_bytes())

    def delete(self, key: str) -> bool:
        path = self._resolve_path(key)
        with self._lock:
            if path.exists():
                path.unlink()
                return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        ext = ".json" if self.use_json else ".pkl"
        keys = []
        for f in self.root.iterdir():
            if f.suffix == ext:
                key = f.stem
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def count(self, prefix: str = "") -> int:
        return len(self.list_keys(prefix))
