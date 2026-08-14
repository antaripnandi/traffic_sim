"""
Phase 1 - 5: Synthetic City Simulation Engine
Generates realistic multi-week urban traffic dynamics with causal relationships:
Population -> Vehicles -> Pressure & Congestion -> Speeds & Violations -> Incidents.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from config import (
    NUM_ZONES,
    NUM_WEEKS,
    RANDOM_SEED,
    POPULATION_MIN,
    POPULATION_MAX,
    ROAD_CAPACITY_MIN,
    ROAD_CAPACITY_MAX,
    WEATHER_PROBABILITIES,
    WEATHER_IMPACT,
    EVENT_PROBABILITIES,
    EVENT_IMPACT,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_city_zones(num_zones: int = NUM_ZONES, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """
    Phase 1: Generates static zone infrastructure properties.
    Each zone has a baseline population density, road network capacity, and zone archetype.
    """
    np.random.seed(seed)
    zones = []
    archetypes = ["Residential", "Commercial", "Transit_Hub", "Mixed_Industrial"]

    for zone_idx in range(1, num_zones + 1):
        zone_id = f"Zone_{zone_idx:02d}"
        population_density = int(np.random.randint(POPULATION_MIN, POPULATION_MAX))
        road_capacity = int(np.random.randint(ROAD_CAPACITY_MIN, ROAD_CAPACITY_MAX))
        archetype = np.random.choice(archetypes, p=[0.40, 0.30, 0.15, 0.15])

        # Commercial/Transit areas attract more vehicles per capita
        vehicle_activity_factor = {
            "Residential": np.random.uniform(0.70, 0.90),
            "Commercial": np.random.uniform(1.10, 1.45),
            "Transit_Hub": np.random.uniform(1.20, 1.60),
            "Mixed_Industrial": np.random.uniform(0.85, 1.15),
        }[archetype]

        zones.append({
            "zone_id": zone_id,
            "archetype": archetype,
            "population_density": population_density,
            "road_capacity": road_capacity,
            "vehicle_activity_factor": vehicle_activity_factor,
        })

    return zones


def simulate_city_traffic(
    num_zones: int = NUM_ZONES,
    num_weeks: int = NUM_WEEKS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Phases 1-5: Generates a full multi-week panel dataset of the synthetic city.
    """
    zones = generate_city_zones(num_zones=num_zones, seed=seed)
    records = []

    weather_types = list(WEATHER_PROBABILITIES.keys())
    weather_probs = list(WEATHER_PROBABILITIES.values())

    event_types = list(EVENT_PROBABILITIES.keys())
    event_probs = list(EVENT_PROBABILITIES.values())

    for week in range(1, num_weeks + 1):
        for zone in zones:
            # ----------------------------------------------------
            # Phase 4: Weather & Special Events
            # ----------------------------------------------------
            weather = np.random.choice(weather_types, p=weather_probs)
            event = np.random.choice(event_types, p=event_probs)

            w_impact = WEATHER_IMPACT[weather]
            e_impact = EVENT_IMPACT[event]

            # ----------------------------------------------------
            # Phase 2: Vehicle Density
            # Influenced by population density, zone archetype, event surge, and weekly variance
            # ----------------------------------------------------
            # Convert population into baseline traffic scale (0 - 150 scale comparable to road capacity)
            base_vehicles = (zone["population_density"] / 100.0) * zone["vehicle_activity_factor"]
            weekly_noise = np.random.uniform(0.90, 1.10)
            vehicle_density = round(base_vehicles * e_impact["veh_mult"] * weekly_noise, 1)

            # Effective road capacity can be restricted by roadworks or heavy rain
            effective_capacity = max(20.0, round(zone["road_capacity"] * e_impact["cap_mult"], 1))

            # ----------------------------------------------------
            # Phase 3: Pressure, Congestion, Speed, Violations
            # ----------------------------------------------------
            # Traffic pressure ratio: demand vs capacity
            pressure_ratio = vehicle_density / effective_capacity

            # Congestion index (0 - 100 scale)
            raw_congestion = 100.0 * sigmoid(2.8 * (pressure_ratio - 1.0)) * w_impact["congestion_mult"]
            congestion = min(100.0, max(5.0, round(raw_congestion, 1)))

            # Average speed (km/h): Free flow speed is 55-60 km/h, reduced nonlinearly by congestion & weather
            free_flow_speed = 58.0
            speed_reduction = (congestion / 100.0) ** 1.35 * 38.0
            speed_noise = np.random.normal(0, 1.5)
            avg_speed = max(10.0, min(65.0, round((free_flow_speed - speed_reduction + speed_noise) * w_impact["speed_mult"], 1)))

            # Traffic violations count: Stop-and-go aggression, speeding in low congestion, or reckless overtaking
            lambda_violations = 3.0 + (congestion / 100.0) * 8.0 + (1.0 if weather != "Clear" else 0.0)
            violations = int(np.random.poisson(max(1.0, lambda_violations)))

            # ----------------------------------------------------
            # Phase 5: Incident Generation (Ground Truth Latent Risk & Occurrence)
            # ----------------------------------------------------
            # Latent risk logit
            risk_logit = (
                -2.80
                + 0.032 * congestion
                + 0.085 * violations
                + 0.040 * max(0.0, 35.0 - avg_speed)
                + w_impact["risk_bias"]
                + e_impact["risk_bias"]
                + np.random.normal(0, 0.15)
            )
            true_risk_prob = float(np.clip(sigmoid(np.array([risk_logit]))[0], 0.02, 0.95))

            # Sample binary incident
            incident_occurred = int(np.random.rand() < true_risk_prob)

            records.append({
                "zone_id": zone["zone_id"],
                "week": week,
                "archetype": zone["archetype"],
                "population_density": zone["population_density"],
                "road_capacity": zone["road_capacity"],
                "effective_capacity": effective_capacity,
                "vehicle_density": vehicle_density,
                "pressure_ratio": round(pressure_ratio, 2),
                "congestion": congestion,
                "avg_speed": avg_speed,
                "violations": violations,
                "weather": weather,
                "event": event,
                "true_risk_prob": round(true_risk_prob, 4),
                "incident_occurred": incident_occurred,
            })

    df = pd.DataFrame(records)
    return df
