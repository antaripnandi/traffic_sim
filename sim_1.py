import numpy as np
import pandas as pd


# ==========================================
# TRAFFIC RISK SIMULATOR
# PHASE 1 + PHASE 2
# ==========================================

NUM_ZONES = 20
NUM_WEEKS = 20

# Makes our results reproducible
np.random.seed(42)


# ==========================================
# PHASE 1
# CREATE BASIC ZONES
# ==========================================

zones = []

for zone_number in range(1, NUM_ZONES + 1):

    zone_id = f"Zone_{zone_number:02d}"

    # Population density:
    # People per square kilometer
    population_density = np.random.randint(1000, 15000)

    # Road capacity:
    # Higher number = road can handle more vehicles
    road_capacity = np.random.randint(40, 150)

    zones.append({
        "zone_id": zone_id,
        "population_density": population_density,
        "road_capacity": road_capacity
    })


# ==========================================
# PHASE 2
# GENERATE VEHICLE DENSITY
# ==========================================

data = []

for zone in zones:

    for week in range(1, NUM_WEEKS + 1):

        population = zone["population_density"]

        # --------------------------------------
        # Convert population into a 0-100 scale
        # --------------------------------------

        population_score = (
            (population - 1000) / (15000 - 1000)
        ) * 100

        # --------------------------------------
        # Create a baseline vehicle density
        # based partly on population
        # --------------------------------------

        base_vehicle_density = population_score * 0.75

        # --------------------------------------
        # Add weekly randomness
        # --------------------------------------

        weekly_variation = np.random.normal(
            loc=0,
            scale=8
        )

        vehicle_density = (
            base_vehicle_density
            + weekly_variation
        )

        # --------------------------------------
        # Keep vehicle density between 0 and 100
        # --------------------------------------

        vehicle_density = np.clip(
            vehicle_density,
            0,
            100
        )

        # --------------------------------------
        # Store the record
        # --------------------------------------

        data.append({
            "zone_id": zone["zone_id"],
            "week": week,
            "population_density": population,
            "road_capacity": zone["road_capacity"],
            "vehicle_density": round(
                vehicle_density,
                2
            )
        })


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(data)


# ==========================================
# DISPLAY DATA
# ==========================================

print("\nPHASE 2 SIMULATION")
print("==================\n")

print(df.head(10))


# ==========================================
# BASIC CHECKS
# ==========================================

print("\nTotal rows:", len(df))

print(
    "Number of zones:",
    df["zone_id"].nunique()
)

print(
    "Number of weeks:",
    df["week"].nunique()
)


# ==========================================
# CHECK VEHICLE DENSITY
# ==========================================

print("\nVehicle density statistics:")

print(
    df["vehicle_density"].describe()
)


# ==========================================
# CHECK POPULATION VS VEHICLES
# ==========================================

print("\nPopulation and vehicle density:")

print(
    df[
        [
            "zone_id",
            "population_density",
            "vehicle_density"
        ]
    ].head(20)
)


# ==========================================
# SAVE DATA
# ==========================================

df.to_csv(
    "../data/phase2_data.csv",
    index=False
)

print("\nPhase 2 dataset saved!")