import os
import numpy as np
import pandas as pd


# ==============================================================================
# TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
# COMPLETE SYNTHETIC CITY SIMULATOR (PHASES 1 TO 5)
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

# Configure Pandas to display ALL rows and ALL columns without truncating
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------------
# HELPER: SIGMOID FUNCTION
# ------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# PHASE 1: CREATE STATIC CITY ZONES
# ==============================================================================
zones = []
archetypes = ["Residential", "Commercial_Downtown", "Transit_Hub", "Mixed_Industrial"]
archetype_probs = [0.40, 0.30, 0.15, 0.15]

for zone_num in range(1, NUM_ZONES + 1):
    zone_id = f"Zone_{zone_num:02d}"
    population_density = int(np.random.randint(POPULATION_MIN, POPULATION_MAX))
    road_capacity = int(np.random.randint(ROAD_CAPACITY_MIN, ROAD_CAPACITY_MAX))
    archetype = np.random.choice(archetypes, p=archetype_probs)

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

        # Phase 4: Weather & Special Events
        weather = "Normal"
        road_condition = "Good"
        special_event = 0
        event_surge = 0.0

        if 10 <= week <= 12 and zone_id in ["Zone_03", "Zone_07", "Zone_12"]:
            special_event = 1
            event_surge = 18.0
            road_condition = "Moderate"

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

        w_congestion_mult = {"Normal": 1.0, "Rain": 1.15, "Heavy Rain": 1.35}[weather]
        w_speed_penalty = {"Normal": 0.0, "Rain": 6.0, "Heavy Rain": 14.0}[weather]
        w_risk_bias = {"Normal": 0.0, "Rain": 0.35, "Heavy Rain": 0.85}[weather]
        road_risk_bias = {"Good": 0.0, "Moderate": 0.25, "Poor": 0.65}[road_condition]

        # Phase 2: Vehicle Density
        pop_score = ((population - POPULATION_MIN) / (POPULATION_MAX - POPULATION_MIN)) * 100.0
        base_vehicles = pop_score * 0.70 * zone["activity_factor"]
        seasonal_drift = 6.0 if (6 <= week <= 9) else 0.0
        weekly_noise = np.random.normal(loc=0.0, scale=7.0)

        vehicle_density = float(np.clip(base_vehicles + event_surge + seasonal_drift + weekly_noise, 0.0, 100.0))

        # Phase 3: Pressure, Congestion, Speed, Violations
        pressure = vehicle_density / max(1.0, capacity)
        raw_congestion = 100.0 * sigmoid(3.2 * (pressure - 0.85)) * w_congestion_mult
        congestion_noise = np.random.normal(0.0, 3.5)
        congestion = float(np.clip(raw_congestion + congestion_noise, 0.0, 100.0))

        speed_drop = (congestion / 100.0) ** 1.3 * 38.0
        raw_speed = 65.0 - speed_drop - w_speed_penalty + np.random.normal(0.0, 2.0)
        average_speed = float(np.clip(raw_speed, 10.0, 80.0))

        expected_violations = 1.0 + (congestion / 100.0) * 8.0 + (2.0 if special_event == 1 else 0.0)
        violations = int(np.random.poisson(lam=max(0.5, expected_violations)))

        # Phase 5: Historical Incidents
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
# CREATE FULL DATAFRAME (400 ROWS = 20 ZONES x 20 WEEKS)
# ==============================================================================
df = pd.DataFrame(records)


# ==============================================================================
# DISPLAY ALL DATA (EVERY SINGLE WEEK AND ZONE)
# ==============================================================================
print("\n" + "=" * 110)
print(" COMPLETE DATASET: ALL 20 WEEKS ACROSS ALL 20 ZONES (TOTAL 400 OBSERVATIONS) ".center(110, "="))
print("=" * 110)

# Print the complete 400-row table
print(df.to_string(index=False))
print("-" * 110)


# ==============================================================================
# 20 x 20 PIVOT GRID (SEE ALL 20 WEEKS AT A GLANCE FOR EACH ZONE)
# ==============================================================================
print("\n" + "=" * 110)
print(" VEHICLE DENSITY MATRIX (ALL 20 ZONES x ALL 20 WEEKS) ".center(110, "="))
print("=" * 110)
pivot_vehicles = df.pivot(index="zone_id", columns="week", values="vehicle_density")
print(pivot_vehicles.to_string())

print("\n" + "=" * 110)
print(" CONGESTION MATRIX (ALL 20 ZONES x ALL 20 WEEKS) ".center(110, "="))
print("=" * 110)
pivot_congestion = df.pivot(index="zone_id", columns="week", values="congestion")
print(pivot_congestion.to_string())

print("\n" + "=" * 110)
print(" INCIDENTS MATRIX (ALL 20 ZONES x ALL 20 WEEKS) [1 = Incident, 0 = No Incident] ".center(110, "="))
print("=" * 110)
pivot_incidents = df.pivot(index="zone_id", columns="week", values="incident_occurred")
print(pivot_incidents.to_string())


# ==============================================================================
# SAVE TO CSV
# ==============================================================================
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, "traffic_simulation.csv")
df.to_csv(output_file, index=False)

print("\n" + "=" * 110)
print(f" [+] Full 400-row dataset successfully saved to: {output_file} ".center(110, "="))
print("=" * 110 + "\n")