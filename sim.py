import os
import numpy as np
import pandas as pd


# ==============================================================================
# TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
# COMPLETE SYNTHETIC CITY SIMULATOR (PHASES 1 TO 5 IN A SINGLE FILE)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. CONFIGURATION CONSTANTS
# ------------------------------------------------------------------------------
NUM_ZONES = 20          # 20 distinct geographical zones
NUM_WEEKS = 20          # 20 consecutive weeks of historical data
RANDOM_SEED = 42        # Ensures reproducible results every time you run

POPULATION_MIN = 1000   # Minimum population per sq km
POPULATION_MAX = 15000  # Maximum population per sq km
ROAD_CAPACITY_MIN = 40  # Minimum baseline road capacity (0-100 scale)
ROAD_CAPACITY_MAX = 100 # Maximum baseline road capacity (0-100 scale)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------------
# HELPER: SIGMOID FUNCTION
# Converts any real number into a smooth probability between 0 and 1
# ------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# PHASE 1: CREATE STATIC CITY ZONES
# ==============================================================================
print("\n" + "=" * 78)
print(" GENERATING SYNTHETIC CITY DATA (20 ZONES x 20 WEEKS = 400 RECORDS) ".center(78, "="))
print("=" * 78)

zones = []
archetypes = ["Residential", "Commercial_Downtown", "Transit_Hub", "Mixed_Industrial"]
archetype_probs = [0.40, 0.30, 0.15, 0.15]

for zone_num in range(1, NUM_ZONES + 1):
    zone_id = f"Zone_{zone_num:02d}"

    # Stable population density: 1,000 to 15,000 people per sq km
    population_density = int(np.random.randint(POPULATION_MIN, POPULATION_MAX))

    # Stable road capacity: 40 to 100
    road_capacity = int(np.random.randint(ROAD_CAPACITY_MIN, ROAD_CAPACITY_MAX))

    # Zone archetype
    archetype = np.random.choice(archetypes, p=archetype_probs)

    # Activity multiplier (Commercial & Transit hubs pull more vehicles per capita)
    activity_factor = {
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
        "activity_factor": activity_factor
    })


# ==============================================================================
# PHASES 2 TO 5: GENERATE MULTI-WEEK PANEL DATA
# ==============================================================================
records = []

for week in range(1, NUM_WEEKS + 1):
    for zone in zones:
        zone_id = zone["zone_id"]
        population = zone["population_density"]
        capacity = zone["road_capacity"]

        # ----------------------------------------------------------------------
        # PHASE 4: EXTERNAL & TEMPORAL CONDITIONS
        # ----------------------------------------------------------------------
        weather = "Normal"
        road_condition = "Good"
        special_event = 0
        event_surge = 0.0

        # Weeks 10-12: Festival/Event traffic surge in select zones
        if 10 <= week <= 12 and zone_id in ["Zone_03", "Zone_07", "Zone_12"]:
            special_event = 1
            event_surge = 18.0
            road_condition = "Moderate"

        # Weeks 16-18: Monsoon / Heavy rain weather shock
        if 16 <= week <= 18:
            roll = np.random.rand()
            if roll < 0.45:
                weather = "Heavy Rain"
                road_condition = "Poor"
            elif roll < 0.85:
                weather = "Rain"
                road_condition = "Moderate"
        else:
            if np.random.rand() < 0.12:
                weather = "Rain"
                road_condition = "Moderate"

        # Weather impacts
        w_congestion_mult = {"Normal": 1.0, "Rain": 1.15, "Heavy Rain": 1.35}[weather]
        w_speed_penalty = {"Normal": 0.0, "Rain": 6.0, "Heavy Rain": 14.0}[weather]
        w_risk_bias = {"Normal": 0.0, "Rain": 0.35, "Heavy Rain": 0.85}[weather]
        road_risk_bias = {"Good": 0.0, "Moderate": 0.25, "Poor": 0.65}[road_condition]

        # ----------------------------------------------------------------------
        # PHASE 2: VEHICLE DENSITY (0 to 100 Scale)
        # ----------------------------------------------------------------------
        pop_score = ((population - POPULATION_MIN) / (POPULATION_MAX - POPULATION_MIN)) * 100.0
        base_vehicles = pop_score * 0.70 * zone["activity_factor"]
        seasonal_drift = 6.0 if (6 <= week <= 9) else 0.0
        weekly_noise = np.random.normal(loc=0.0, scale=7.0)

        vehicle_density = float(np.clip(base_vehicles + event_surge + seasonal_drift + weekly_noise, 0.0, 100.0))

        # ----------------------------------------------------------------------
        # PHASE 3: TRAFFIC PRESSURE, CONGESTION, SPEED, VIOLATIONS
        # ----------------------------------------------------------------------
        # Pressure ratio: Demand (Vehicles) / Supply (Road Capacity)
        pressure = vehicle_density / max(1.0, capacity)

        # Congestion (0-100): Sigmoid curve on traffic pressure
        raw_congestion = 100.0 * sigmoid(3.2 * (pressure - 0.85)) * w_congestion_mult
        congestion_noise = np.random.normal(0.0, 3.5)
        congestion = float(np.clip(raw_congestion + congestion_noise, 0.0, 100.0))

        # Average Speed (10-80 km/h): Free flow (65 km/h) reduced by congestion & weather
        speed_drop = (congestion / 100.0) ** 1.3 * 38.0
        raw_speed = 65.0 - speed_drop - w_speed_penalty + np.random.normal(0.0, 2.0)
        average_speed = float(np.clip(raw_speed, 10.0, 80.0))

        # Red Light Violations: Poisson distribution driven by congestion
        expected_violations = 1.0 + (congestion / 100.0) * 8.0 + (2.0 if special_event == 1 else 0.0)
        violations = int(np.random.poisson(lam=max(0.5, expected_violations)))

        # ----------------------------------------------------------------------
        # PHASE 5: HISTORICAL INCIDENTS (LATENT RISK & TARGET GENERATION)
        # ----------------------------------------------------------------------
        risk_logit = (
            -3.30
            + 0.038 * congestion
            + 0.110 * violations
            + 0.045 * max(0.0, 45.0 - average_speed)
            + w_risk_bias
            + road_risk_bias
            + np.random.normal(0.0, 0.20)
        )

        incident_prob = float(np.clip(sigmoid(risk_logit), 0.02, 0.95))
        incident_occurred = int(np.random.rand() < incident_prob)
        incident_count = int(1 + np.random.poisson(lam=incident_prob * 1.2)) if incident_occurred else 0

        # Store full record
        records.append({
            "zone_id": zone_id,
            "week": week,
            "population_density": population,
            "road_capacity": capacity,
            "vehicle_density": round(vehicle_density, 2),
            "congestion": round(congestion, 2),
            "average_speed": round(average_speed, 2),
            "red_light_violations": violations,
            "weather": weather,
            "road_condition": road_condition,
            "special_event": special_event,
            "incident_count": incident_count,
            "incident_occurred": incident_occurred,
        })


# ==============================================================================
# CREATE FINAL DATAFRAME
# ==============================================================================
df = pd.DataFrame(records)


# ==============================================================================
# VALIDATION & DIAGNOSTIC REPORT
# ==============================================================================
print("\n[+] 1. DATASET DIMENSIONS:")
print(f"    - Total Rows:        {len(df)} (Expected: {NUM_ZONES * NUM_WEEKS})")
print(f"    - Unique Zones:      {df['zone_id'].nunique()} (Zone_01 to Zone_{NUM_ZONES:02d})")
print(f"    - Unique Weeks:      {df['week'].nunique()} (Week 1 to Week {NUM_WEEKS})")

print("\n[+] 2. EARLY WEEKS PREVIEW (Week 1 - First 5 Zones):")
print("-" * 78)
print(df[df["week"] == 1].head(5).to_string(index=False))
print("-" * 78)

print("\n[+] 3. LATE WEEKS PREVIEW (Week 20 - Last 5 Zones):")
print("-" * 78)
print(df[df["week"] == 20].tail(5).to_string(index=False))
print("-" * 78)

print("\n[+] 4. COMPLETE 20-WEEK TIMELINE FOR ZONE_03:")
print("-" * 78)
zone_trace = df[df["zone_id"] == "Zone_03"][["zone_id", "week", "vehicle_density", "congestion", "average_speed", "weather", "incident_occurred"]]
print(zone_trace.to_string(index=False))
print("-" * 78)

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

print("\n[+] 6. CAUSAL CORRELATION MATRIX:")
num_cols = ["population_density", "vehicle_density", "road_capacity", "congestion", "average_speed", "red_light_violations", "incident_occurred"]
print(df[num_cols].corr().round(3).to_string())

print("\n[+] 7. DATA QUALITY & RANGE ASSERTIONS:")
v_min, v_max = df["vehicle_density"].min(), df["vehicle_density"].max()
c_min, c_max = df["congestion"].min(), df["congestion"].max()
s_min, s_max = df["average_speed"].min(), df["average_speed"].max()
inc_rate = df["incident_occurred"].mean() * 100.0

print(f"    - Vehicle Density Range:   {v_min:.1f} to {v_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= v_min and v_max <= 100 else 'FAIL'})")
print(f"    - Congestion Range:        {c_min:.1f} to {c_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= c_min and c_max <= 100 else 'FAIL'})")
print(f"    - Average Speed Range:     {s_min:.1f} to {s_max:.1f} km/h (Valid [10, 80]: {'PASS' if 10 <= s_min and s_max <= 80 else 'FAIL'})")
print(f"    - Overall Incident Rate:   {inc_rate:.1f}% (Healthy realistic baseline for ML)")


# ==============================================================================
# SAVE DATASETS TO CSV
# ==============================================================================
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

output_file = os.path.join(data_dir, "traffic_simulation.csv")
df.to_csv(output_file, index=False)
print(f"\n[+] Full dataset ({len(df)} rows) successfully saved to:\n    {output_file}\n")
print("=" * 78 + "\n")
