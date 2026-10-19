"""Convert complete SIP daily demand records to the environment's wide CSV format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


RAW_COLUMNS = ["day", "O", "D", "t", "demand"]


def _convert_one_day(raw_path, output_path, raw_day, zone_count, timestep_count):
    frame = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
    expected_rows = zone_count * zone_count * timestep_count
    if len(frame) != expected_rows:
        raise ValueError(f"{raw_path.name}: expected {expected_rows} rows, found {len(frame)}")
    if frame[RAW_COLUMNS].isna().any().any():
        raise ValueError(f"{raw_path.name}: missing values are not allowed")
    if set(frame["day"].astype(int)) != {raw_day}:
        raise ValueError(f"{raw_path.name}: day column must contain only {raw_day}")
    if (frame["demand"] < 0).any() or ((frame["demand"] % 1) != 0).any():
        raise ValueError(f"{raw_path.name}: demand must be nonnegative integers")

    expected_keys = pd.MultiIndex.from_product(
        [range(zone_count), range(zone_count), range(timestep_count)],
        names=["O", "D", "t"],
    )
    actual_keys = pd.MultiIndex.from_frame(frame[["O", "D", "t"]].astype(int))
    if actual_keys.has_duplicates or not actual_keys.equals(expected_keys):
        raise ValueError(f"{raw_path.name}: incomplete, duplicate, or unsorted O-D-time grid")
    if (frame.loc[frame["O"] == frame["D"], "demand"] != 0).any():
        raise ValueError(f"{raw_path.name}: intra-zone demand cannot be represented by this environment")

    interzone = frame.loc[frame["O"] != frame["D"], ["O", "D", "t", "demand"]].copy()
    wide = interzone.pivot(index=["O", "D"], columns="t", values="demand")
    wide = wide.reindex(columns=range(timestep_count)).reset_index()
    wide.columns = ["O", "D", *[str(t) for t in range(timestep_count)]]
    wide = wide.astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    wide.to_csv(temporary, index=False)
    os.replace(temporary, output_path)


def convert_demand_directory(
    raw_dir,
    output_dir,
    *,
    day_count=30,
    zone_count=50,
    timestep_count=26,
):
    """Convert q_0..q_(day_count-1) into one-based od_demand_1..N files."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    outputs = []
    for raw_day in range(day_count):
        raw_path = raw_dir / f"q_{raw_day}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        output_path = output_dir / f"od_demand_{raw_day + 1}.csv"
        _convert_one_day(raw_path, output_path, raw_day, zone_count, timestep_count)
        outputs.append(output_path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = convert_demand_directory(args.raw_dir, args.output_dir)
    print(f"Converted {len(outputs)} complete SIP demand days into {args.output_dir}")


if __name__ == "__main__":
    main()
