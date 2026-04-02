"""用于测试 轻量 存储后端."""

import json

from infra.storage import ExperienceJsonlLogger, JsonlLogger
from memory.experience_pool import ExperienceRecord


def test_jsonl_logger_writes_event(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlLogger(path)
    logger.log({"event": "state", "generation": 1})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["generation"] == 1


def test_experience_jsonl_logger_writes_record(tmp_path) -> None:
    path = tmp_path / "experience.jsonl"
    logger = ExperienceJsonlLogger(path)
    logger.log(
        ExperienceRecord(
            state={"generation": 2},
            action={"generation": 2, "mutation_prob": 0.2, "crossover_prob": 0.85, "reason": "test"},
            reward=0.5,
            next_state={"generation": 3},
        )
    )

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["reward"] == 0.5
    assert payload["next_state"]["generation"] == 3
