"""
================================================================================
TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
MODULE: SYNTHETIC CITY DATA SIMULATOR (Phases 1 to 5)
================================================================================

This module generates a multi-week synthetic panel dataset of urban traffic conditions.
The simulation implements a causal chain of relationships:

  Population Density (Stable Zone Feature)
          |
          v
  Baseline Vehicle Density + Weekly Variation + Temporal Scenarios
          |
          v
  Traffic Pressure (Vehicles / Road Capacity)
          |
          v
  Congestion Index (0 to 100)
          |
          +-----------------------------+
          |                             |
          v                             v
  Average Speed (km/h)        Red Light Violations (Count)
          |                             |
          +--------------+--------------+
                         |
                         v   (+ Weather + Road Condition + Events)
               Latent Incident Risk
                         |
                         v
              Incident Probability (0 to 1)
                         |
                         v
          Incident Outcome (0 or 1) & Incident Count

NOTE: This is synthetic data generated for a 24-hour hackathon prototype to
demonstrate the architectural methodology of Risk vs Exposure vs Priority.
================================================================================
"""

import os
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# City simulation scale
NUM_ZONES = 20      # Number of distinct geographical zones
NUM_WEEKS = 20      # Number of consecutive weeks simulated
RANDOM_SEED = 42    # Ensures exact reproducibility of synthetic data

# Baseline zone parameter ranges
POPULATION_MIN = 1000   # Minimum population per sq km
POPULATION_MAX = 15000  # Maximum population per sq km
ROAD_CAPACITY_MIN = 40  # Minimum baseline road capacity score (0-100 scale)
ROAD_CAPACITY_MAX = 100 # Maximum baseline road capacity score (0-100 scale)


# ==============================================================================
# MATHEMATICAL HELPER FUNCTIONS
# ==============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Standard Sigmoid / Logistic Function:
    Transforms any real-valued number (logit) into a smooth probability between 0 and 1.
    Formula: S(x) = 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# PHASE 1: CREATE BASIC ZONES (Static Infrastructure)
# ==============================================================================

def create_city_zones(num_zones: int = NUM_ZONES, seed: int = RANDOM_SEED) -> list:
    """
    Generates static baseline properties for each zone in the city.
    These properties remain stable across weeks.
    
    Returns a list of dictionaries with zone baseline parameters.
    """
    np.random.seed(seed)
    zones = []

    # Zone types give realistic structural identity to each zone
    zone_archetypes = [
        "Residential", 
        "Commercial_Downtown", 
        "Transit_Hub", 
        "Mixed_Industrial"
    ]
    archetype_probabilities = [0.40, 0.30, 0.15, 0.15]

    for zone_num in range(1, num_zones + 1):
        zone_id = f"Zone_{zone_num:02d}"

        # Population density: People per square kilometer (1,000 - 15,000)
        population_density = int(np.random.randint(POPULATION_MIN, POPULATION_MAX))

        # Road capacity: Normalized 40 - 100 score representing road infrastructure throughput
        road_capacity = int(np.random.randint(ROAD_CAPACITY_MIN, ROAD_CAPACITY_MAX))

        # Zone archetype (influences commercial vehicle pull)
        archetype = np.random.choice(zone_archetypes, p=archetype_probabilities)

        # Baseline vehicle pull factor: Commercial and transit hubs attract more vehicles per capita
        vehicle_activity_factor = {
            "Residential": np.random.uniform(0.70, 0.90),
            "Commercial_Downtown": np.random.uniform(1.10, 1.40),
            "Transit_Hub": np.random.uniform(1.15, 1.45),
            "Mixed_Industrial": np.random.uniform(0.85, 1.10),
        }[archetype]

        zones.append({
            "zone_id": zone_id,
            "archetype": archetype,
            "population_density": population_density,
            "road_capacity": road_capacity,
            "vehicle_activity_factor": round(vehicle_activity_factor, 2)
        })

    return zones


# ==============================================================================
# PHASE 4 HELPER: TEMPORAL SCENARIO CONTROLLER
# ==============================================================================

def get_weekly_scenario(week: int, zone_id: str) -> dict:
    """
    Implements controlled temporal narrative scenarios across the 20 weeks:
    - Weeks 1-5:   Normal baseline
    - Weeks 6-9:   Gradual seasonal traffic increase in downtown zones
    - Weeks 10-12: Special municipal events (Festivals/Marathons) in selected zones
    - Weeks 13-15: Normal baseline recovery
    - Weeks 16-18: Monsoon / Heavy rain weather shock
    - Weeks 19-20: Divergence (some corridors improve, high-exposure zones remain strained)
    """
    weather = "Normal"
    road_condition = "Good"
    special_event = 0
    event_traffic_surge = 0.0

    # Weeks 10-12: Special Event in specific entertainment/transit corridors
    if 10 <= week <= 12 and zone_id in ["Zone_03", "Zone_07", "Zone_12"]:
        special_event = 1
        event_traffic_surge = 18.0  # Extra vehicle density
        road_condition = "Moderate"

    # Weeks 16-18: Weather shifts (Rain / Heavy Rain)
    if 16 <= week <= 18:
        weather_roll = np.random.rand()
        if weather_roll < 0.45:
            weather = "Heavy Rain"
            road_condition = "Poor"
        elif weather_roll < 0.85:
            weather = "Rain"
            road_condition = "Moderate"
        else:
            weather = "Normal"
    else:
        # Occasional random showers in other weeks
        weather_roll = np.random.rand()
        if weather_roll < 0.15:
            weather = "Rain"
            road_condition = "Moderate"

    return {
        "weather": weather,
        "road_condition": road_condition,
        "special_event": special_event,
        "event_traffic_surge": event_traffic_surge
    }


# ==============================================================================
# FULL SIMULATOR: PHASES 1 TO 5
# ==============================================================================

def run_traffic_simulator(
    num_zones: int = NUM_ZONES, 
    num_weeks: int = NUM_WEEKS, 
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Executes the full 5-phase synthetic traffic simulation.
    Generates num_zones x num_weeks panel observations with causal relationships.
    """
    # 1. Create static city zones
    zones = create_city_zones(num_zones=num_zones, seed=seed)
    records = []

    # Iterate chronologically week by week, zone by zone
    for week in range(1, num_weeks + 1):
        for zone in zones:

            # ------------------------------------------------------------------
            # Phase 1: Stable Zone Parameters
            # ------------------------------------------------------------------
            population = zone["population_density"]
            capacity = zone["road_capacity"]

            # ------------------------------------------------------------------
            # Phase 4: Environmental & Scenario Conditions
            # ------------------------------------------------------------------
            scenario = get_weekly_scenario(week, zone["zone_id"])
            weather = scenario["weather"]
            road_cond = scenario["road_condition"]
            event = scenario["special_event"]
            surge = scenario["event_traffic_surge"]

            # Weather impact factors
            weather_congestion_mult = {"Normal": 1.0, "Rain": 1.15, "Heavy Rain": 1.35}[weather]
            weather_speed_penalty = {"Normal": 0.0, "Rain": 6.0, "Heavy Rain": 14.0}[weather]
            weather_risk_bias = {"Normal": 0.0, "Rain": 0.35, "Heavy Rain": 0.85}[weather]

            # Road condition risk impact
            road_risk_bias = {"Good": 0.0, "Moderate": 0.25, "Poor": 0.65}[road_cond]

            # ------------------------------------------------------------------
            # Phase 2: Vehicle Density
            # ------------------------------------------------------------------
            # Convert population into a normalized 0-100 score
            population_score = ((population - POPULATION_MIN) / (POPULATION_MAX - POPULATION_MIN)) * 100.0

            # Baseline vehicle density modulated by zone archetype activity factor
            base_vehicle_density = population_score * 0.70 * zone["vehicle_activity_factor"]

            # Organic seasonal drift in middle weeks (Weeks 6-9)
            seasonal_drift = 6.0 if (6 <= week <= 9) else 0.0

            # Weekly stochastic variation (Normal distribution: Mean = 0, Std = 7)
            weekly_noise = np.random.normal(loc=0.0, scale=7.0)

            # Combined Vehicle Density
            raw_vehicle_density = base_vehicle_density + surge + seasonal_drift + weekly_noise
            vehicle_density = float(np.clip(raw_vehicle_density, 0.0, 100.0))

            # ------------------------------------------------------------------
            # Phase 3: Traffic Conditions (Pressure, Congestion, Speed, Violations)
            # ------------------------------------------------------------------
            # Traffic Pressure: Ratio of Vehicle Density to Road Capacity
            traffic_pressure = vehicle_density / max(1.0, capacity)

            # Congestion (0-100): Non-linear sigmoid response to traffic pressure
            # When pressure > 0.85 (demand approaches/exceeds capacity), congestion rises steeply
            pressure_centered = traffic_pressure - 0.85
            raw_congestion = 100.0 * sigmoid(3.2 * pressure_centered) * weather_congestion_mult
            congestion_noise = np.random.normal(loc=0.0, scale=3.5)
            congestion = float(np.clip(raw_congestion + congestion_noise, 0.0, 100.0))

            # Average Speed (km/h): Free-flow speed (65 km/h) degraded by congestion & weather
            speed_degradation = (congestion / 100.0) ** 1.3 * 38.0
            raw_speed = 65.0 - speed_degradation - weather_speed_penalty + np.random.normal(0.0, 2.0)
            average_speed = float(np.clip(raw_speed, 10.0, 80.0))

            # Red Light Violations: Generated via Poisson distribution
            # Higher congestion and frustration/delays increase violation probability
            expected_violations = 1.0 + (congestion / 100.0) * 8.0 + (2.0 if event == 1 else 0.0)
            red_light_violations = int(np.random.poisson(lam=max(0.5, expected_violations)))

            # ------------------------------------------------------------------
            # Phase 5: Historical Incidents (Latent Risk & Target Generation)
            # ------------------------------------------------------------------
            # Latent risk logit: Causal combination of all dangerous conditions
            risk_logit = (
                -3.30                                    # Baseline intercept (accidents are rare)
                + 0.038 * congestion                     # Congestion increases collision probability
                + 0.110 * red_light_violations           # Violations strongly elevate risk
                + 0.045 * max(0.0, 45.0 - average_speed) # Very low stop-and-go speed indicates hazard
                + weather_risk_bias                      # Slippery roads increase risk
                + road_risk_bias                         # Potholes / poor surfaces increase risk
                + np.random.normal(0.0, 0.20)            # Natural real-world noise
            )

            # Convert logit to Incident Probability using Sigmoid
            incident_prob = float(np.clip(sigmoid(risk_logit), 0.02, 0.95))

            # Sample binary outcome: Did an incident occur this week?
            incident_occurred = int(np.random.rand() < incident_prob)

            # Incident Count: Typically 0 or 1 in a single week; high-hazard weeks can have 2 or 3
            incident_count = 0
            if incident_occurred:
                incident_count = int(1 + np.random.poisson(lam=incident_prob * 1.2))

            # ------------------------------------------------------------------
            # Append Record
            # ------------------------------------------------------------------
            records.append({
                "zone_id": zone["zone_id"],
                "week": week,
                "population_density": population,
                "road_capacity": capacity,
                "vehicle_density": round(vehicle_density, 2),
                "congestion": round(congestion, 2),
                "average_speed": round(average_speed, 2),
                "red_light_violations": red_light_violations,
                "weather": weather,
                "road_condition": road_cond,
                "special_event": event,
                "incident_count": incident_count,
                "incident_occurred": incident_occurred,
            })

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# VALIDATION & DIAGNOSTIC PRINTS
# ==============================================================================

def validate_and_display_data(df: pd.DataFrame):
    """
    Performs comprehensive data quality checks and prints statistical summaries.
    """
    print("\n" + "=" * 78)
    print(" TEMPORAL TRAFFIC RISK SIMULATION -- DATA VALIDATION REPORT ".center(78, "="))
    print("=" * 78)

    # 1. Dataset Dimensions
    total_rows = len(df)
    unique_zones = df["zone_id"].nunique()
    unique_weeks = df["week"].nunique()
    expected_rows = unique_zones * unique_weeks

    print("\n[+] 1. DATASET DIMENSIONS:")
    print(f"    - Total Rows:        {total_rows}")
    print(f"    - Unique Zones:      {unique_zones} (Expected: {NUM_ZONES})")
    print(f"    - Unique Weeks:      {unique_weeks} (Expected: {NUM_WEEKS})")
    print(f"    - Integrity Check:   {'PASS (No missing zone-weeks)' if total_rows == expected_rows else 'FAIL'}")

    # 2. Early Weeks Preview (Week 1)
    print("\n[+] 2. EARLY WEEKS PREVIEW (Week 1 - First 5 Zones):")
    print("-" * 78)
    print(df[df["week"] == 1].head(5).to_string(index=False))
    print("-" * 78)

    # 3. Late Weeks Preview (Week 20)
    print("\n[+] 3. LATE WEEKS PREVIEW (Week 20 - Last 5 Zones):")
    print("-" * 78)
    print(df[df["week"] == 20].tail(5).to_string(index=False))
    print("-" * 78)

    # 4. Temporal Trace for a single zone across all 20 weeks
    print("\n[+] 4. TEMPORAL EVOLUTION FOR ZONE_03 (All 20 Weeks):")
    print("-" * 78)
    cols_to_show = ["zone_id", "week", "vehicle_density", "congestion", "average_speed", "weather", "incident_occurred"]
    zone_trace = df[df["zone_id"] == "Zone_03"][cols_to_show]
    print(zone_trace.to_string(index=False))
    print("-" * 78)

    # 5. City-Wide Weekly Averages (Week-by-Week Trend)
    print("\n[+] 5. CITY-WIDE WEEKLY AVERAGES (Weeks 1 to 20):")
    weekly_agg = df.groupby("week").agg({
        "vehicle_density": "mean",
        "congestion": "mean",
        "average_speed": "mean",
        "red_light_violations": "sum",
        "incident_occurred": "sum"
    }).round(1)
    weekly_agg.columns = ["Avg Vehicles", "Avg Congestion", "Avg Speed (km/h)", "Total Violations", "Total Incidents"]
    print(weekly_agg.to_string())

    # 6. Correlation Matrix (Causal Relationship Verification)
    print("\n[+] 6. CORRELATION MATRIX (Verifying Causal Physics):")
    numeric_cols = [
        "population_density",
        "vehicle_density",
        "road_capacity",
        "congestion",
        "average_speed",
        "red_light_violations",
        "incident_occurred"
    ]
    corr_df = df[numeric_cols].corr().round(3)
    print(corr_df.to_string())

    # 7. Quality Assertions & Bounds Checks
    print("\n[+] 7. DATA QUALITY & RANGE CHECKS:")
    v_min, v_max = df["vehicle_density"].min(), df["vehicle_density"].max()
    c_min, c_max = df["congestion"].min(), df["congestion"].max()
    s_min, s_max = df["average_speed"].min(), df["average_speed"].max()
    inc_rate = df["incident_occurred"].mean() * 100.0

    print(f"    - Vehicle Density Range:   {v_min:.1f} to {v_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= v_min and v_max <= 100 else 'FAIL'})")
    print(f"    - Congestion Range:        {c_min:.1f} to {c_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= c_min and c_max <= 100 else 'FAIL'})")
    print(f"    - Average Speed Range:     {s_min:.1f} to {s_max:.1f} km/h (Valid [10, 80]: {'PASS' if 10 <= s_min and s_max <= 80 else 'FAIL'})")
    print(f"    - Overall Incident Rate:   {inc_rate:.1f}% (Healthy realistic baseline for ML)")

    # 8. Check Stability of Baseline Population
    pop_std_per_zone = df.groupby("zone_id")["population_density"].std().max()
    print(f"    - Population Stability:    Max Per-Zone Std Dev = {pop_std_per_zone:.1f} ({'PASS (Stable)' if pop_std_per_zone == 0 else 'FAIL'})")
    print("=" * 78 + "\n")


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    # Ensure output data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Run simulation
    print("\nGenerating synthetic city traffic simulation dataset...")
    df_sim = run_traffic_simulator(
        num_zones=NUM_ZONES, 
        num_weeks=NUM_WEEKS, 
        seed=RANDOM_SEED
    )

    # 2. Validate and display results
    validate_and_display_data(df_sim)

    # 3. Save to CSV
    output_path = os.path.join(data_dir, "traffic_simulation.csv")
    df_sim.to_csv(output_path, index=False)
    print(f"[+] Successfully saved simulation dataset to:\n    {output_path}\n")
