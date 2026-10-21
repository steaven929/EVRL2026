"""Atomic per-day evaluation output for reviewer experiments."""

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


EXPLORATION_HISTORY_COLUMNS = (
    ("exploration_base_sigma_history", "base_sigma"),
    ("exploration_service_ema_history", "service_rate_ema"),
    ("exploration_service_gap_history", "service_gap"),
    ("exploration_boost_history", "service_boost"),
    ("exploration_sigma_history", "final_sigma"),
)

CHARGING_EVENT_COLUMNS = (
    "test_day",
    "timestep",
    "vehicle_id",
    "event",
    "origin_zone",
    "target_zone",
    "zone",
    "charger_id",
    "travel_distance_internal",
    "soc_before_internal",
    "soc_after_internal",
    "grid_energy_kwh",
    "battery_energy_kwh",
    "charging_cost_rmb",
)

TRAINING_CHARGING_EVENT_COLUMNS = (
    "episode",
    "timestep",
    "demand_day",
    *CHARGING_EVENT_COLUMNS[2:],
)

TRAINING_CHARGING_SUMMARY_COLUMNS = (
    "episode",
    "demand_day",
    "charging_grid_energy_kwh",
    "charging_battery_energy_kwh",
    "charging_energy_cost",
    "charging_travel_distance_km",
    "charging_travel_cost",
    "charging_completion_count",
    "charging_reservation_count",
    "charging_event_count",
)


@dataclass(frozen=True)
class EvaluationIdentity:
    method: str
    fleet_size: int
    seed: int
    charger_ratio: float
    test_day: int


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> Sequence[str]:
    names = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                names.append(str(key))
    return names


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] = (),
) -> None:
    materialized = [dict(row) for row in rows]
    detected_names = _ordered_fieldnames(materialized)
    ordered_names = list(fieldnames)
    ordered_names.extend(name for name in detected_names if name not in ordered_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            if ordered_names:
                writer = csv.DictWriter(handle, fieldnames=ordered_names)
                writer.writeheader()
                writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_exploration_history(output_dir, result_data: Mapping[str, Any]) -> None:
    """Persist every exploration driver used by the reviewer E2/E3 runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for result_key, _ in EXPLORATION_HISTORY_COLUMNS:
        if result_key not in result_data:
            continue
        array = np.asarray(result_data[result_key], dtype=float)
        arrays[result_key] = array
        np.save(output_dir / f"{result_key}.npy", array)

    if not arrays:
        return
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("exploration history arrays must have equal lengths")
    row_count = lengths.pop()
    rows = []
    for index in range(row_count):
        row = {"episode": index + 1}
        for result_key, csv_name in EXPLORATION_HISTORY_COLUMNS:
            if result_key in arrays:
                row[csv_name] = float(arrays[result_key][index])
        rows.append(row)
    _write_csv(output_dir / "exploration_history.csv", rows)


def write_records_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] = (),
) -> None:
    """Atomically persist arbitrary experiment records."""
    _write_csv(Path(path), rows, fieldnames=fieldnames)


class EvaluationWriter:
    def __init__(self, output_root: Path, experiment: str):
        self.root = Path(output_root) / str(experiment)
        self.root.mkdir(parents=True, exist_ok=True)

    def _day_dir(self, identity: EvaluationIdentity) -> Path:
        return (
            self.root
            / identity.method
            / str(identity.fleet_size)
            / str(identity.seed)
            / f"ratio_{identity.charger_ratio:.2f}"
            / f"day_{identity.test_day}"
        )

    def write_day(
        self,
        identity: EvaluationIdentity,
        config: Mapping[str, Any],
        summary: Mapping[str, Any],
        timestep_rows: Iterable[Mapping[str, Any]],
        zone_timestep_rows: Iterable[Mapping[str, Any]],
        charging_event_rows: Iterable[Mapping[str, Any]],
        reward_rows: Iterable[Mapping[str, Any]],
        solver_rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        day_dir = self._day_dir(identity)
        day_dir.mkdir(parents=True, exist_ok=True)
        identity_fields = asdict(identity)
        _write_json(day_dir / "config.json", {**dict(config), **identity_fields})
        _write_json(day_dir / "summary.json", {**identity_fields, **dict(summary)})
        _write_csv(day_dir / "timestep_metrics.csv", timestep_rows)
        _write_csv(day_dir / "zone_timestep_metrics.csv", zone_timestep_rows)
        _write_csv(
            day_dir / "vehicle_charging_events.csv",
            charging_event_rows,
            fieldnames=CHARGING_EVENT_COLUMNS,
        )
        _write_csv(day_dir / "reward_components.csv", reward_rows)
        _write_csv(day_dir / "solver_stats.csv", solver_rows)
        self._mark_complete(identity, summary, day_dir)
        return day_dir

    def _mark_complete(
        self,
        identity: EvaluationIdentity,
        summary: Mapping[str, Any],
        day_dir: Path,
    ) -> None:
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"episodes": []}
        record = {
            **asdict(identity),
            "complete": True,
            "path": str(day_dir.relative_to(self.root)),
            "summary": dict(summary),
        }
        key = (
            identity.method,
            identity.fleet_size,
            identity.seed,
            identity.charger_ratio,
            identity.test_day,
        )
        retained = []
        for item in manifest.get("episodes", []):
            item_key = (
                item.get("method"),
                item.get("fleet_size"),
                item.get("seed"),
                item.get("charger_ratio"),
                item.get("test_day"),
            )
            if item_key != key:
                retained.append(item)
        retained.append(record)
        retained.sort(
            key=lambda item: (
                str(item["method"]),
                int(item["fleet_size"]),
                int(item["seed"]),
                float(item["charger_ratio"]),
                int(item["test_day"]),
            )
        )
        manifest["episodes"] = retained
        _write_json(manifest_path, manifest)
        summary_rows = []
        for item in retained:
            summary_rows.append(
                {
                    "method": item["method"],
                    "fleet_size": item["fleet_size"],
                    "seed": item["seed"],
                    "charger_ratio": item["charger_ratio"],
                    "test_day": item["test_day"],
                    "complete": item["complete"],
                    **item.get("summary", {}),
                }
            )
        _write_csv(self.root / "evaluation_summary.csv", summary_rows)
