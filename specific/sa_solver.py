import numpy as np


def _normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 0.0:
        size = max(1, arr.size)
        return np.full(arr.shape, 1.0 / size, dtype=np.float32)
    return (arr / total).astype(np.float32)


def build_seed_action(demand_out, available, charger_free):
    dispatch_from = _normalize(available)
    dispatch_to = _normalize(demand_out)
    charge_pref = _normalize(np.asarray(charger_free, dtype=np.float32) + 1e-6)
    return np.concatenate([dispatch_from, dispatch_to, charge_pref]).astype(np.float32)


class SimulatedAnnealingSolver:
    def __init__(self, zone_num, init_temp, final_temp, cooling_rate, iterations, seed=None, rng=None):
        self.zone_num = int(zone_num)
        self.init_temp = float(init_temp)
        self.final_temp = float(final_temp)
        self.cooling_rate = float(cooling_rate)
        self.iterations = int(iterations)

        if self.zone_num <= 0:
            raise ValueError("zone_num must be positive")
        if self.init_temp <= 0.0 or self.final_temp <= 0.0:
            raise ValueError("temperatures must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not (0.0 < self.cooling_rate < 1.0):
            raise ValueError("cooling_rate must be in (0, 1)")

        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def propose_neighbor(self, action):
        neighbor = np.asarray(action, dtype=np.float32).copy()
        idx = int(self.rng.integers(0, neighbor.size))
        delta = float(self.rng.uniform(-0.15, 0.15))
        neighbor[idx] = np.clip(neighbor[idx] + delta, 0.0, 1.0)

        z = self.zone_num
        neighbor[:z] = _normalize(neighbor[:z])
        neighbor[z : 2 * z] = _normalize(neighbor[z : 2 * z])
        neighbor[2 * z : 3 * z] = _normalize(neighbor[2 * z : 3 * z])
        return neighbor.astype(np.float32)

    def score_action(self, action, snapshot):
        action = np.asarray(action, dtype=np.float32)
        z = self.zone_num
        expected_dim = 3 * z
        if action.shape != (expected_dim,):
            raise ValueError(f"action shape {action.shape} != ({expected_dim},)")

        dispatch_from = action[:z]
        dispatch_to = action[z : 2 * z]
        charge_pref = action[2 * z : 3 * z]

        demand_out = np.asarray(snapshot["demand_out"], dtype=np.float32)
        demand_in = np.asarray(snapshot["demand_in"], dtype=np.float32)
        available = np.asarray(snapshot["available"], dtype=np.float32)
        avg_soc = np.asarray(snapshot["avg_soc"], dtype=np.float32)
        charger_free = np.asarray(snapshot["charger_free"], dtype=np.float32)
        distance = np.asarray(snapshot["distance"], dtype=np.float32)

        weighted_distance = np.mean(distance, axis=1) if distance.ndim == 2 else np.zeros(z, dtype=np.float32)
        service_gain = float(np.dot(dispatch_to, demand_out + 0.35 * demand_in))
        dispatch_support = float(np.dot(dispatch_from, np.minimum(available, demand_out + 1.0)))
        dispatch_cost = float(np.dot(dispatch_from, weighted_distance + 0.2 * np.maximum(demand_in - demand_out, 0.0)))
        low_soc_penalty = float(np.dot(dispatch_from, np.clip(0.35 - avg_soc, 0.0, None)))
        charging_bonus = float(np.dot(charge_pref, charger_free + np.clip(0.55 - avg_soc, 0.0, None)))
        unmet_penalty = float(np.sum(np.clip(demand_out - dispatch_to * np.sum(available), 0.0, None)))

        return (
            service_gain
            + 0.4 * dispatch_support
            + 0.7 * charging_bonus
            - 0.15 * dispatch_cost
            - 3.0 * low_soc_penalty
            - 0.1 * unmet_penalty
        )

    def solve(self, snapshot):
        current = build_seed_action(
            demand_out=snapshot["demand_out"],
            available=snapshot["available"],
            charger_free=snapshot["charger_free"],
        )
        current_score = self.score_action(current, snapshot)
        best = current.copy()
        best_score = current_score
        temperature = self.init_temp

        while temperature > self.final_temp:
            for _ in range(self.iterations):
                candidate = self.propose_neighbor(current)
                candidate_score = self.score_action(candidate, snapshot)
                delta = candidate_score - current_score
                if delta >= 0.0 or float(self.rng.random()) < np.exp(delta / max(temperature, 1e-6)):
                    current = candidate
                    current_score = candidate_score
                    if current_score > best_score:
                        best = current.copy()
                        best_score = current_score
            temperature *= self.cooling_rate

        return np.clip(best, 0.0, 1.0).astype(np.float32)
