"""Experience pool 基础组件 用于 M5 记忆-enabled 闭环 控制."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ExperienceRecord:
    """从一次控制周期中捕获的单条转移元组."""

    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    next_state: dict[str, Any]
    # Optional dynamic-context fields for DWTA scripted stages/waves.
    stage_id: str | None = None
    wave_id: str | None = None
    trigger_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化 该 记录 用于 存储后端."""
        return asdict(self)


class ExperiencePool:
    """内存中的滑动窗口存储 用于 recent 控制 experiences.

    Uses ``collections.deque(maxlen=N)`` for O(1) automatic eviction
    instead of list slicing which allocates a new list on every overflow.
    """

    def __init__(self, max_size: int = 100) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        self._records: deque[ExperienceRecord] = deque(maxlen=max_size)

    def append(self, record: ExperienceRecord) -> None:
        """添加一条经验; deque 自动丢弃最旧记录."""
        self._records.append(record)

    def recent(self, n: int) -> list[ExperienceRecord]:
        """返回 at most 该 last ``n`` records 在 时间顺序."""
        if n <= 0:
            return []
        # deque doesn't support negative slicing; convert via islice
        start = max(0, len(self._records) - n)
        return list(self._records)[start:]

    def get_recent(self, n: int) -> list[ExperienceRecord]:
        """为未来调用方兼容性保留的别名."""
        return self.recent(n)

    def __len__(self) -> int:
        return len(self._records)

