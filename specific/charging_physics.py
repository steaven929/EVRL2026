"""Physical-unit conversions and deterministic charger allocation."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence, Tuple


@dataclass(frozen=True)
class ChargingConfig:
    distance_unit_km: float
    battery_capacity_kwh: float
    battery_capacity_internal: float
    energy_consumption_kwh_per_km: float
    charging_power_kw: float
    charging_efficiency: float
    soc_min_ratio: float
    soc_target_ratio: float
    electricity_price_rmb_per_kwh: float

    def __post_init__(self) -> None:
        positive = {
            "distance_unit_km": self.distance_unit_km,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "battery_capacity_internal": self.battery_capacity_internal,
            "energy_consumption_kwh_per_km": self.energy_consumption_kwh_per_km,
            "charging_power_kw": self.charging_power_kw,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.charging_efficiency <= 1:
            raise ValueError("charging_efficiency must be in (0, 1]")
        if not 0 <= self.soc_min_ratio < self.soc_target_ratio <= 1:
            raise ValueError("SOC ratios must satisfy 0 <= min < target <= 1")
        if self.electricity_price_rmb_per_kwh < 0:
            raise ValueError("electricity_price_rmb_per_kwh cannot be negative")

    @property
    def internal_per_kwh(self) -> float:
        return self.battery_capacity_internal / self.battery_capacity_kwh

    @property
    def minimum_soc_internal(self) -> float:
        return self.battery_capacity_internal * self.soc_min_ratio

    @property
    def target_soc_internal(self) -> float:
        return self.battery_capacity_internal * self.soc_target_ratio


@dataclass(frozen=True)
class ChargingUpdate:
    soc_before_internal: float
    soc_after_internal: float
    grid_energy_kwh: float
    battery_energy_kwh: float
    charging_cost_rmb: float
    effective_charging_minutes: float
    reached_target: bool


def distance_to_energy_kwh(distance_internal: float, config: ChargingConfig) -> float:
    distance = max(0.0, float(distance_internal))
    return distance * config.distance_unit_km * config.energy_consumption_kwh_per_km


def energy_kwh_to_internal(energy_kwh: float, config: ChargingConfig) -> float:
    return max(0.0, float(energy_kwh)) * config.internal_per_kwh


def internal_to_energy_kwh(soc_internal: float, config: ChargingConfig) -> float:
    return max(0.0, float(soc_internal)) / config.internal_per_kwh


def charging_duration_minutes(soc_internal: float, config: ChargingConfig) -> float:
    current = min(max(0.0, float(soc_internal)), config.target_soc_internal)
    required_battery_kwh = internal_to_energy_kwh(
        config.target_soc_internal - current, config
    )
    if required_battery_kwh <= 0:
        return 0.0
    grid_power_to_battery_kw = config.charging_power_kw * config.charging_efficiency
    return required_battery_kwh / grid_power_to_battery_kw * 60.0


def charge_for_duration(
    soc_internal: float,
    charging_minutes: float,
    config: ChargingConfig,
) -> ChargingUpdate:
    before = min(max(0.0, float(soc_internal)), config.battery_capacity_internal)
    available_minutes = max(0.0, float(charging_minutes))
    if before >= config.target_soc_internal - 1e-9:
        return ChargingUpdate(
            soc_before_internal=before,
            soc_after_internal=before,
            grid_energy_kwh=0.0,
            battery_energy_kwh=0.0,
            charging_cost_rmb=0.0,
            effective_charging_minutes=0.0,
            reached_target=True,
        )
    required_minutes = charging_duration_minutes(before, config)
    effective_minutes = min(available_minutes, required_minutes)
    grid_energy = config.charging_power_kw * effective_minutes / 60.0
    battery_energy = grid_energy * config.charging_efficiency
    after = min(
        config.target_soc_internal,
        before + energy_kwh_to_internal(battery_energy, config),
    )
    return ChargingUpdate(
        soc_before_internal=before,
        soc_after_internal=after,
        grid_energy_kwh=grid_energy,
        battery_energy_kwh=battery_energy,
        charging_cost_rmb=grid_energy * config.electricity_price_rmb_per_kwh,
        effective_charging_minutes=effective_minutes,
        reached_target=after >= config.target_soc_internal - 1e-9,
    )


def rounded_charger_count(total_parking: int, ratio: float) -> int:
    if total_parking < 0:
        raise ValueError("total_parking cannot be negative")
    if ratio < 0:
        raise ValueError("ratio cannot be negative")
    raw = Decimal(str(total_parking)) * Decimal(str(ratio))
    return int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def allocate_chargers_by_demand(
    total_chargers: int,
    parking_capacity: Sequence[int],
    demand_score: Sequence[float],
    ensure_zone_coverage: bool = False,
) -> Tuple[int, ...]:
    if len(parking_capacity) != len(demand_score):
        raise ValueError("parking_capacity and demand_score must have equal length")
    if total_chargers < 0 or total_chargers > sum(parking_capacity):
        raise ValueError("total_chargers is outside total parking capacity")
    if any(value < 0 for value in parking_capacity):
        raise ValueError("parking capacity cannot be negative")

    allocation = [0] * len(parking_capacity)
    remaining = total_chargers
    if ensure_zone_coverage:
        coverable = [i for i, capacity in enumerate(parking_capacity) if capacity > 0]
        if total_chargers < len(coverable):
            raise ValueError("not enough chargers to cover every zone")
        for zone in coverable:
            allocation[zone] = 1
        remaining -= len(coverable)

    ranking = sorted(
        range(len(parking_capacity)),
        key=lambda zone: (-float(demand_score[zone]), zone),
    )
    while remaining:
        assigned_this_round = False
        for zone in ranking:
            if allocation[zone] < parking_capacity[zone]:
                allocation[zone] += 1
                remaining -= 1
                assigned_this_round = True
                if remaining == 0:
                    break
        if not assigned_this_round:
            raise ValueError("unable to allocate chargers within parking capacity")
    return tuple(allocation)
