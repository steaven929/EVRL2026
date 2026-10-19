"""Shared output location for every training entrypoint."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "resluts2610"


def result_run_directory(run_name):
    """Return one run directory under the user-selected results root."""
    run_name = str(run_name).strip()
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one non-empty folder name")
    return RESULTS_ROOT / run_name
