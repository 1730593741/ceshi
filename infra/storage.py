"""轻量存储工具 用于 实验 事件 与 experiences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory.experience_pool import ExperienceRecord


class JsonlLogger:
    """追加写入式 JSONL 日志器 用于 状态/动作 事件.

    Uses a persistent file handle with explicit flush to avoid repeated
    open/close system calls on every log() invocation (significant I/O
    overhead on Windows NTFS with hundreds of generations).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        """将一个事件写成单行 JSON 对象."""
        self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Release the underlying file handle."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        self.close()


class ExperienceJsonlLogger:
    """持久化 experience tuples 到 JSONL 用于 离线分析.

    Uses a persistent file handle with explicit flush for the same
    I/O performance reasons as JsonlLogger.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, record: ExperienceRecord) -> None:
        """Write 一个 ``ExperienceRecord`` 作为 一个 JSONL line."""
        self._file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Release the underlying file handle."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        self.close()


class CsvExporter:
    """占位接口 用于 未来 CSV 导出扩展."""

    def export(self, rows: list[dict[str, Any]], path: str | Path) -> None:
        """未来扩展点 用于 轻量 CSV 输出."""
        raise NotImplementedError("CSV export is not implemented in M5 MVP")
