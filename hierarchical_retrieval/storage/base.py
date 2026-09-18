"""
存储层抽象接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class StorageBase(ABC):
    """所有存储后端的统一抽象"""

    @abstractmethod
    def save(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        ...

    @abstractmethod
    def count(self, prefix: str = "") -> int:
        ...