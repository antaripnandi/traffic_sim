"""
================================================================================
TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
MODULE: UPGRADED SYNTHETIC CITY SIMULATOR (Phases 1 to 5)
================================================================================

This module generates a rich, realistic multi-week synthetic panel dataset of
urban traffic conditions specifically designed for temporal ML risk modeling
and multi-factor priority decision systems.

CORE PROJECT SEPARATION:
1. RISK (Probability)       : P(Incident | Conditions, Physics, Weather, History)
2. EXPOSURE (Volume)        : Population Density + Vehicle Density (Citizens at risk)
3. PRIORITY (Action Index)  : Where intervention delivers the greatest impact

DATA ARCHITECTURE:
- Base Variables            : zone_id, week, zone_type, population_density, road_capacity
- Simulated Variables       : vehicle_density, weather, road_condition, special_event
- Derived Traffic Physics   : effective_road_capacity, traffic_pressure, congestion, average_speed, red_light_violations
- Target Variables          : incident_count, incident_occurred

================================================================================
"""

import os
import numpy as np
import pandas as pd


# ==============================================================================
# 0. CONFIGURATION CONSTANTS (Easily Scalable)
# ==============================================================================

NUM_ZONES = 50          # 50 distinct geographical urban zones
NUM_WEEKS = 52          # 52 weeks (1 full year of temporal data = 2,600 observations)
RANDOM_SEED = 42        # Ensures exact mathematical reproducibility

# Zone baseline bounds
POPULATION_MIN = 1200   # Min residential population per sq km
POPULATION_MAX = 16000  # Max residential population per sq km
ROAD_CAPACITY_MIN = 40  # Min road infrastructure capacity score (0-100 scale)
ROAD_CAPACITY_MAX = 100 # Max road infrastructure capacity score (0-100 scale)


# ==============================================================================
# 1. MATHEMATICAL HELPER FUNCTIONS
# ==============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Standard Logistic Sigmoid Function:
    Smoothly maps any real-valued number (logit) into a bounded probability [0, 1].
    Formula: S(x) = 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# 2. PHASE 1: ZONE GENERATOR & ARCHETYPES (Stable Infrastructure)
# ==============================================================================

def create_city_zones(num_zones: int = NUM_ZONES, seed: int = RANDOM_SEED) -> list:
    """
    Creates persistent physical zones with realistic urban personalities.
    
    Zone Types & Behavior:
    - Residential         : High population, moderate commuter traffic
    - Commercial_Downtown : Moderate population, extremely high daytime vehicle pull
    - University_District : High footfall, highly sensitive to semester events
    - Industrial_Corridor : Low population, heavy freight traffic, steady demand
    - Highway_Junction    : Very low population, massive vehicular throughput
    - Suburban_LowDensity : Low population, low traffic volume
    
    Zone Trajectories (Micro-Trends):
    - Worsening           : Chronic degradation / construction bottleneck
    - Improving           : Signal retiming / gradual decongestion
    - Stable              : Predictable baseline
    - Event_Sensitive     : Sharp spiky surges
    """
    np.random.seed(seed)
    zones = []

    archetypes = [
        "Residential",
        "Commercial_Downtown",
        "University_District",
        "Industrial_Corridor",
        "Highway_Junction",
        "Suburban_LowDensity"
    ]
    archetype_probs = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]

    trend_types = ["Stable", "Worsening", "Improving", "Event_Sensitive"]
    trend_probs = [0.45, 0.25, 0.15, 0.15]

    for zone_num in range(1, num_zones + 1):
        zone_id = f"Zone_{zone_num:02d}"
        archetype = np.random.choice(archetypes, p=archetype_probs)
        trend_profile = np.random.choice(trend_types, p=trend_probs)

        # Baseline population tailored to archetype
        if archetype == "Residential":
            pop = np.random.randint(8000, POPULATION_MAX)
            activity_pull = np.random.uniform(0.70, 0.85)
        elif archetype == "Commercial_Downtown":
            pop = np.random.randint(3500, 9500)
            activity_pull = np.random.uniform(1.20, 1.55)  # Heavy commuter pull
        elif archetype == "University_District":
            pop = np.random.randint(6000, 12000)
            activity_pull = np.random.uniform(1.00, 1.30)
        elif archetype == "Industrial_Corridor":
            pop = np.random.randint(1800, 4500)
            activity_pull = np.random.uniform(1.10, 1.35)  # Freight trucks
        elif archetype == "Highway_Junction":
            pop = np.random.randint(POPULATION_MIN, 3000)  # Very low population
            activity_pull = np.random.uniform(1.40, 1.80)  # Massive traffic corridor
        else: # Suburban_LowDensity
            pop = np.random.randint(POPULATION_MIN, 4500)
            activity_pull = np.random.uniform(0.50, 0.70)

        # Baseline road capacity (40 to 100)
        base_capacity = int(np.random.randint(ROAD_CAPACITY_MIN, ROAD_CAPACITY_MAX))

        zones.append({
            "zone_id": zone_id,
            "zone_type": archetype,
            "trend_profile": trend_profile,
            "population_density": int(pop),
            "road_capacity": base_capacity,
            "activity_pull": round(activity_pull, 2)
        })

    return zones


# ==============================================================================
# 3. PHASE 4: ENVIRONMENTAL & TEMPORAL SCENARIO CONTROLLER
# ==============================================================================

def get_temporal_environment(week: int, zone: dict) -> dict:
    """
    Implements macro city-wide seasonal timeline across 52 weeks and
    micro zone-specific special events:
    
    Macro Timeline:
    - Weeks 1-8   : Q1 Baseline (Standard winter/early spring traffic)
    - Weeks 9-16  : Q1-Q2 Organic Growth (City-wide economic and commuter rise)
    - Weeks 17-20 : Spring Festival & Marathon Season (Targeted zone surges)
    - Weeks 21-30 : Summer Steady Flow
    - Weeks 31-36 : Monsoon / Severe Rain Season (Weather shocks & capacity drops)
    - Weeks 37-44 : Autumn Major Conventions & Concerts
    - Weeks 45-52 : Year-End Peak Holiday Rush & Winter Divergence
    """
    zone_id = zone["zone_id"]
    archetype = zone["zone_type"]
    trend_profile = zone["trend_profile"]

    # 1. Weather Modeling
    weather = "Normal"
    road_condition = "Good"

    # Monsoon Season (Weeks 31-36)
    if 31 <= week <= 36:
        roll = np.random.rand()
        if roll < 0.35:
            weather = "Heavy Rain"
            road_condition = "Poor"
        elif roll < 0.75:
            weather = "Light Rain"
            road_condition = "Moderate"
    # Winter Rain / Storms (Weeks 48-52)
    elif 48 <= week <= 52:
        if np.random.rand() < 0.40:
            weather = "Light Rain"
            road_condition = "Moderate"
    else:
        # Sporadic showers during normal periods
        if np.random.rand() < 0.12:
            weather = "Light Rain"
            road_condition = "Moderate"

    # 2. Special Events Modeling (Festivals, Concerts, Marathons, Stadium Games)
    special_event = 0
    event_surge_vehicles = 0.0
    event_capacity_penalty = 1.0

    # Spring Festival (Weeks 18-19 in Commercial and University districts)
    if (18 <= week <= 19) and (archetype in ["Commercial_Downtown", "University_District"]):
        special_event = 1
        event_surge_vehicles = 16.0
        event_capacity_penalty = 0.85  # Street closures for pedestrians

    # Autumn Concerts & Conventions (Weeks 38-40 in Highway Junctions and Downtown)
    elif (38 <= week <= 40) and (archetype in ["Commercial_Downtown", "Highway_Junction"]):
        special_event = 1
        event_surge_vehicles = 22.0
        event_capacity_penalty = 0.80

    # Zone-Specific Event-Sensitive recurring spikes
    elif trend_profile == "Event_Sensitive" and (week in [7, 14, 25, 43]):
        special_event = 1
        event_surge_vehicles = 18.0
        event_capacity_penalty = 0.88

    # 3. Zone-Specific Micro Trend Delta (Evolution over 52 weeks)
    if trend_profile == "Worsening":
        # Gradual infrastructure wear / increasing bottleneck (+0.25 vehicles per week)
        trend_drift = (week / 52.0) * 14.0
    elif trend_profile == "Improving":
        # Gradual corridor improvement (-0.20 vehicles per week)
        trend_drift = -(week / 52.0) * 10.0
    else:
        trend_drift = 0.0

    return {
        "weather": weather,
        "road_condition": road_condition,
        "special_event": special_event,
        "event_surge_vehicles": event_surge_vehicles,
        "event_capacity_penalty": event_capacity_penalty,
        "trend_drift": trend_drift
    }


# ==============================================================================
# 4. PHYSICAL DYNAMICS & SIMULATION FUNCTIONS
# ==============================================================================

def calculate_effective_capacity(
    base_capacity: int,
    weather: str,
    road_condition: str,
    event_capacity_penalty: float
) -> float:
    """
    Upgrade 5 & 6: Computes usable effective road throughput.
    Effective Capacity = Base Capacity * Weather Factor * Road Condition Factor * Event Disruption
    """
    w_factor = {"Normal": 1.00, "Light Rain": 0.92, "Heavy Rain": 0.78}[weather]
    r_factor = {"Good": 1.00, "Moderate": 0.90, "Poor": 0.75}[road_condition]
    
    effective_cap = base_capacity * w_factor * r_factor * event_capacity_penalty
    return max(15.0, round(effective_cap, 2))


def calculate_vehicle_density(
    population_density: int,
    activity_pull: float,
    event_surge: float,
    trend_drift: float,
    weather: str,
    prev_density: float = None
) -> float:
    """
    Upgrade 4: Computes vehicle density (0-100 scale).
    Combines:
    - Base Activity pull from population
    - Zone micro-trend drift
    - Special event influx
    - Autoregressive continuity (this week is anchored to last week + noise)
    """
    pop_score = ((population_density - POPULATION_MIN) / (POPULATION_MAX - POPULATION_MIN)) * 100.0
    base_target = pop_score * 0.65 * activity_pull + event_surge + trend_drift

    # Weather dampening (Heavy rain slightly reduces non-essential travel)
    if weather == "Heavy Rain":
        base_target *= 0.92

    # Autoregressive smoothing (AR(1) process: 65% previous week + 35% new target + noise)
    if prev_density is not None:
        target = 0.65 * prev_density + 0.35 * base_target
    else:
        target = base_target

    weekly_noise = np.random.normal(loc=0.0, scale=4.5)
    return float(np.clip(round(target + weekly_noise, 2), 0.0, 100.0))


def calculate_congestion(
    vehicle_density: float,
    effective_capacity: float,
    weather: str
) -> tuple:
    """
    Upgrade 5 & 7: Computes Traffic Pressure and Congestion Index (0-100 scale).
    Traffic Pressure = Vehicle Density / Effective Road Capacity
    Congestion follows a non-linear Sigmoid response curve to pressure.
    """
    traffic_pressure = vehicle_density / max(1.0, effective_capacity)

    # Sigmoid curve centered around pressure threshold 0.85
    w_mult = {"Normal": 1.00, "Light Rain": 1.12, "Heavy Rain": 1.28}[weather]
    raw_congestion = 100.0 * sigmoid(3.4 * (traffic_pressure - 0.85)) * w_mult
    noise = np.random.normal(0.0, 2.5)

    congestion = float(np.clip(round(raw_congestion + noise, 2), 0.0, 100.0))
    return round(traffic_pressure, 3), congestion


def calculate_speed(congestion: float, weather: str, road_condition: str) -> float:
    """
    Upgrade 8: Computes average vehicle speed (km/h).
    Free-flow speed (65 km/h) reduced non-linearly by congestion and surface conditions.
    """
    w_penalty = {"Normal": 0.0, "Light Rain": 5.0, "Heavy Rain": 13.0}[weather]
    r_penalty = {"Good": 0.0, "Moderate": 3.0, "Poor": 8.0}[road_condition]

    speed_degradation = (congestion / 100.0) ** 1.35 * 38.0
    raw_speed = 65.0 - speed_degradation - w_penalty - r_penalty + np.random.normal(0.0, 1.8)
    return float(np.clip(round(raw_speed, 2), 10.0, 80.0))


def generate_violations(congestion: float, special_event: int, archetype: str) -> int:
    """
    Upgrade 9: Computes Red-Light / Traffic Violations count.
    Modeled as a Poisson count process driven by congestion and corridor type.
    """
    type_bias = 2.0 if archetype in ["Commercial_Downtown", "Highway_Junction"] else 0.0
    event_bias = 3.0 if special_event == 1 else 0.0
    
    expected_lambda = 1.0 + (congestion / 100.0) * 9.0 + type_bias + event_bias
    violations = int(np.random.poisson(lam=max(0.5, expected_lambda)))
    return violations


def generate_incidents(
    congestion: float,
    violations: int,
    average_speed: float,
    traffic_pressure: float,
    weather: str,
    road_condition: str,
    recent_incident_memory: float
) -> tuple:
    """
    Upgrade 15, 16 & 17: Generates Incident Probability, Occurrence, and Count.
    Uses a latent log-odds function combining causal physics and historical memory.
    """
    w_risk = {"Normal": 0.0, "Light Rain": 0.30, "Heavy Rain": 0.80}[weather]
    r_risk = {"Good": 0.0, "Moderate": 0.25, "Poor": 0.60}[road_condition]

    # Latent Risk Logit
    risk_logit = (
        -3.40                                    # Baseline intercept
        + 0.036 * congestion                     # Congestion impact
        + 0.095 * violations                     # Violations impact
        + 0.040 * max(0.0, 42.0 - average_speed) # Stop-and-go speed hazard
        + 0.350 * max(0.0, traffic_pressure - 1.0) # Capacity overflow penalty
        + 0.250 * recent_incident_memory         # Autoregressive historical memory
        + w_risk
        + r_risk
        + np.random.normal(0.0, 0.18)            # Natural real-world variance
    )

    incident_prob = float(np.clip(sigmoid(risk_logit), 0.01, 0.96))
    
    # Stochastic Bernoulli outcome
    incident_occurred = int(np.random.rand() < incident_prob)
    
    # Incident count (Poisson count given incident occurred)
    incident_count = 0
    if incident_occurred:
        incident_count = int(1 + np.random.poisson(lam=incident_prob * 1.3))

    return incident_count, incident_occurred


# ==============================================================================
# 5. FULL SIMULATION PIPELINE (Phases 1 to 5)
# ==============================================================================

def run_traffic_simulator(
    num_zones: int = NUM_ZONES,
    num_weeks: int = NUM_WEEKS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Executes the full upgraded synthetic traffic simulation across all zones and weeks.
    """
    zones = create_city_zones(num_zones=num_zones, seed=seed)
    records = []

    # Historical state tracker for autoregressive memory (zone_id -> dict of past values)
    history_tracker = {
        z["zone_id"]: {
            "prev_density": None,
            "past_incidents": []
        } for z in zones
    }

    # Chronological simulation week-by-week
    for week in range(1, num_weeks + 1):
        for zone in zones:
            z_id = zone["zone_id"]
            pop = zone["population_density"]
            base_cap = zone["road_capacity"]
            archetype = zone["zone_type"]
            activity_pull = zone["activity_pull"]

            # 1. Environmental & Scenario conditions
            env = get_temporal_environment(week, zone)
            weather = env["weather"]
            road_cond = env["road_condition"]
            event = env["special_event"]
            event_surge = env["event_surge_vehicles"]
            cap_penalty = env["event_capacity_penalty"]
            trend_drift = env["trend_drift"]

            # 2. Effective Road Capacity
            eff_capacity = calculate_effective_capacity(
                base_capacity=base_cap,
                weather=weather,
                road_condition=road_cond,
                event_capacity_penalty=cap_penalty
            )

            # 3. Vehicle Density
            prev_dens = history_tracker[z_id]["prev_density"]
            vehicle_density = calculate_vehicle_density(
                population_density=pop,
                activity_pull=activity_pull,
                event_surge=event_surge,
                trend_drift=trend_drift,
                weather=weather,
                prev_density=prev_dens
            )
            history_tracker[z_id]["prev_density"] = vehicle_density

            # 4. Traffic Pressure & Congestion
            pressure, congestion = calculate_congestion(
                vehicle_density=vehicle_density,
                effective_capacity=eff_capacity,
                weather=weather
            )

            # 5. Average Speed
            avg_speed = calculate_speed(
                congestion=congestion,
                weather=weather,
                road_condition=road_cond
            )

            # 6. Red Light Violations
            violations = generate_violations(
                congestion=congestion,
                special_event=event,
                archetype=archetype
            )

            # 7. Historical Incident Memory (Strictly past weeks, zero data leakage)
            past_inc = history_tracker[z_id]["past_incidents"]
            recent_memory = float(np.mean(past_inc[-4:])) if len(past_inc) > 0 else 0.0

            # 8. Incident Generation
            inc_count, inc_occurred = generate_incidents(
                congestion=congestion,
                violations=violations,
                average_speed=avg_speed,
                traffic_pressure=pressure,
                weather=weather,
                road_condition=road_cond,
                recent_incident_memory=recent_memory
            )
            history_tracker[z_id]["past_incidents"].append(inc_occurred)

            # Append complete record
            records.append({
                "zone_id": z_id,
                "week": week,
                "zone_type": archetype,
                "population_density": pop,
                "road_capacity": base_cap,
                "effective_road_capacity": eff_capacity,
                "vehicle_density": vehicle_density,
                "traffic_pressure": pressure,
                "congestion": congestion,
                "average_speed": avg_speed,
                "red_light_violations": violations,
                "weather": weather,
                "road_condition": road_cond,
                "special_event": event,
                "incident_count": inc_count,
                "incident_occurred": inc_occurred,
            })

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# 6. DATA QUALITY VALIDATION & DIAGNOSTIC REPORT
# ==============================================================================

def validate_and_display_data(df: pd.DataFrame):
    """
    Executes automated data quality validations, causal correlation verification,
    and displays structured diagnostic previews.
    """
    print("\n" + "=" * 90)
    print(" TEMPORAL TRAFFIC SIMULATOR -- AUTOMATED VALIDATION REPORT ".center(90, "="))
    print("=" * 90)

    # 1. Dimensions
    total_rows = len(df)
    unique_zones = df["zone_id"].nunique()
    unique_weeks = df["week"].nunique()
    expected_rows = unique_zones * unique_weeks

    print("\n[+] 1. DATASET DIMENSIONS:")
    print(f"    - Total Records Generated: {total_rows:,}")
    print(f"    - Unique Zones:            {unique_zones}")
    print(f"    - Consecutive Weeks:       {unique_weeks}")
    print(f"    - Grid Completeness:       {'PASS (Zero missing zone-weeks)' if total_rows == expected_rows else 'FAIL'}")
    print(f"    - Null Value Check:        {'PASS (0 null values)' if df.isnull().sum().sum() == 0 else 'FAIL'}")

    # 2. Preview (First 5 Rows)
    print("\n[+] 2. DATASET SAMPLE PREVIEW:")
    print("-" * 90)
    sample_cols = [
        "zone_id", "week", "zone_type", "population_density", "vehicle_density",
        "effective_road_capacity", "traffic_pressure", "congestion", "average_speed",
        "red_light_violations", "weather", "special_event", "incident_occurred"
    ]
    print(df[sample_cols].head(6).to_string(index=False))
    print("-" * 90)

    # 3. Zone Personalities Verification (Proving Decoupling of Pop vs Vehicles)
    print("\n[+] 3. ZONE PERSONALITY DECOUPLING AUDIT (Mean per Zone Type):")
    zone_type_summary = df.groupby("zone_type").agg({
        "population_density": "mean",
        "vehicle_density": "mean",
        "congestion": "mean",
        "incident_occurred": "mean"
    }).round(1)
    zone_type_summary["incident_occurred"] = (zone_type_summary["incident_occurred"] * 100).round(1).astype(str) + "%"
    zone_type_summary.columns = ["Avg Population", "Avg Vehicles", "Avg Congestion", "Incident Rate"]
    print(zone_type_summary.to_string())

    # 4. Causal Correlation Matrix
    print("\n[+] 4. CAUSAL CORRELATION MATRIX (Verifying System Physics):")
    numeric_cols = [
        "population_density", "vehicle_density", "effective_road_capacity",
        "traffic_pressure", "congestion", "average_speed", "red_light_violations",
        "incident_occurred"
    ]
    corr_matrix = df[numeric_cols].corr().round(3)
    print(corr_matrix.to_string())

    # 5. Data Range Checks
    print("\n[+] 5. BOUNDS & RANGE VERIFICATION:")
    v_min, v_max = df["vehicle_density"].min(), df["vehicle_density"].max()
    c_min, c_max = df["congestion"].min(), df["congestion"].max()
    s_min, s_max = df["average_speed"].min(), df["average_speed"].max()
    inc_rate = df["incident_occurred"].mean() * 100.0

    print(f"    - Vehicle Density Range:   {v_min:.1f} to {v_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= v_min and v_max <= 100 else 'FAIL'})")
    print(f"    - Congestion Range:        {c_min:.1f} to {c_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= c_min and c_max <= 100 else 'FAIL'})")
    print(f"    - Speed Range:             {s_min:.1f} to {s_max:.1f} km/h (Valid [10, 80]: {'PASS' if 10 <= s_min and s_max <= 80 else 'FAIL'})")
    print(f"    - Overall Incident Rate:   {inc_rate:.1f}% (Healthy non-trivial ML signal)")
    print("=" * 90 + "\n")


# ==============================================================================
# 7. MAIN EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    # Ensure data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir if os.path.basename(script_dir) == "traffic_sim" else os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n[+] Executing Upgraded Simulation ({NUM_ZONES} Zones x {NUM_WEEKS} Weeks = {NUM_ZONES * NUM_WEEKS:,} records)...")
    df_sim = run_traffic_simulator(
        num_zones=NUM_ZONES,
        num_weeks=NUM_WEEKS,
        seed=RANDOM_SEED
    )

    # Validate output
    validate_and_display_data(df_sim)

    # Save dataset
    output_path = os.path.join(data_dir, "traffic_simulation.csv")
    df_sim.to_csv(output_path, index=False)
    print(f"[+] Successfully saved upgraded simulation dataset to:\n    {output_path}\n")
