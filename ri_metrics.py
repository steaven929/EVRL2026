import numpy as np


def _mean_and_last(values):
    if values is None:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite)), float(finite[-1])


def build_ri_exports(episode_step_metrics=None, rld_history=None, rld_last_history=None, by_timestep=None, timestep_mean=None):
    if rld_history is None:
        episode_step_metrics = episode_step_metrics or []
        ri_history = []
        ri_last_history = []
        for episode_values in episode_step_metrics:
            ri_mean, ri_last = _mean_and_last(episode_values)
            ri_history.append(ri_mean)
            ri_last_history.append(ri_last)
    else:
        ri_history = list(rld_history)
        if rld_last_history is None:
            ri_last_history = list(rld_history)
        else:
            ri_last_history = list(rld_last_history)

    return {
        'ri_history': ri_history,
        'ri_last_history': ri_last_history,
        'ri_by_timestep': by_timestep,
        'ri_timestep_mean': timestep_mean,
    }
