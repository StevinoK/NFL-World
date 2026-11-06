from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np

from ridge_game_sim.paths import canonical_hash, output_dir, write_json


def json_value(value):
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def task_hash(task):
    return canonical_hash(json_value({key: value for key, value in task.items() if key not in {"task_hash", "data_dir"}}))


def validate_result(task, result):
    for key in ("task_hash", "bundle_hash", "game_id", "run_id", "week"):
        if result.get(key) != task[key]:
            raise ValueError(f"Chunk handshake mismatch: {key}")
    wanted = sorted(task["simulation_ids"])
    actual = [int(row["simulation"]) for row in result["game_rows"]]
    if sorted(actual) != wanted or sorted(result["completed_ids"]) != wanted or result["completed_count"] != len(wanted):
        raise ValueError("Incomplete/duplicate simulation IDs or counts")
    for row in result["game_rows"]:
        if row.get("game_id") != task["game_id"] or row.get("home_team") != task["home_team"] or row.get("away_team") != task["away_team"]:
            raise ValueError("Game row identity mismatch")
        if not all(np.isfinite(row[key]) for key in ("home_score", "away_score")):
            raise ValueError("Nonfinite game scores")
    for collection in ("player_rows", "diagnostics", "bucket_rows", "nan_rows"):
        for row in result.get(collection, []):
            if row.get("game_id") != task["game_id"] or row.get("simulation") not in wanted:
                raise ValueError(f"Invalid {collection} identity")
    if sorted(row["simulation"] for row in result["diagnostics"]) != wanted:
        raise ValueError("Missing diagnostics")


class ChunkStore:
    def __init__(self, directory: Path, identity: dict, resume=False):
        self.directory = output_dir(directory)
        self.chunks = output_dir(directory / "chunks")
        path = directory / "run_manifest.json"
        if path.exists():
            if not resume:
                raise ValueError("Run already exists; use --resume")
            if json.loads(path.read_text())["identity"] != identity:
                raise ValueError("Resume bundle/settings changed")
        else:
            write_json(path, {"identity": identity, "status": "prepared"})
        self.identity = identity
        self.completed = {}
        for path in sorted(self.chunks.glob("*.json")):
            record = json.loads(path.read_text())
            if task_hash(record["task"]) != record["task"]["task_hash"]:
                raise ValueError("Persisted task hash changed")
            if canonical_hash(record["result"]) != record["result_hash"]:
                raise ValueError("Persisted result changed")
            validate_result(record["task"], record["result"])
            if record["task"]["bundle_hash"] != identity["bundle_hash"] or record["task"]["run_id"] != identity["run_id"]:
                raise ValueError("Chunk belongs to a different run")
            for row in record["result"]["game_rows"]:
                key = (row["game_id"], row["simulation"])
                if key in self.completed:
                    raise ValueError("Duplicate persisted simulation")
                self.completed[key] = record["task"]["task_hash"]

    def save(self, task, result):
        validate_result(task, result)
        if task_hash(task) != task["task_hash"]:
            raise ValueError("Task hash changed")
        path = self.chunks / f"{task['task_hash']}.json"
        record = json_value({"task": task, "result": result})
        record["result_hash"] = canonical_hash(record["result"])
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError("Conflicting deterministic retry")
            return
        if any((task["game_id"], simulation) in self.completed for simulation in task["simulation_ids"]):
            raise ValueError("Overlapping chunk results")
        write_json(path, record)
        for simulation in task["simulation_ids"]:
            self.completed[(task["game_id"], simulation)] = task["task_hash"]

    def results(self):
        for path in sorted(self.chunks.glob("*.json")):
            yield json.loads(path.read_text())["result"]