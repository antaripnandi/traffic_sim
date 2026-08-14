import numpy as np
import pandas as pd


# ==========================================
# PHASE 1: CREATE BASIC TRAFFIC ZONES
# ==========================================

NUM_ZONES = 20
NUM_WEEKS = 20

np.random.seed(42)


# ------------------------------------------
# 1. Create the zones
# ------------------------------------------

zones = []

for zone_number in range(1, NUM_ZONES + 1):

    zone_id = f"Zone_{zone_number:02d}"

    # Population density:
    # People per square kilometer
    population_density = np.random.randint(1000, 15000)

    # Road capacity:
    # Approximate number of vehicles the road
    # network can handle during the observation period
    road_capacity = np.random.randint(40, 150)

    zones.append({
        "zone_id": zone_id,
        "population_density": population_density,
        "road_capacity": road_capacity
    })


# ------------------------------------------
# 2. Create weekly records
# ------------------------------------------

data = []

for zone in zones:

    for week in range(1, NUM_WEEKS + 1):

        data.append({
            "zone_id": zone["zone_id"],
            "week": week,
            "population_density": zone["population_density"],
            "road_capacity": zone["road_capacity"]
        })


# ------------------------------------------
# 3. Convert to DataFrame
# ------------------------------------------

df = pd.DataFrame(data)


# ------------------------------------------
# 4. Display the result
# ------------------------------------------

print("\nPHASE 1 SIMULATION")
print("==================\n")

print(df.head(10))

print("\nTotal rows:", len(df))

print("\nNumber of zones:", df["zone_id"].nunique())

print("\nNumber of weeks:", df["week"].nunique())


# ------------------------------------------
# 5. Check population distribution
# ------------------------------------------

print("\nPopulation density statistics:")
print(df["population_density"].describe())


# ------------------------------------------
# 6. Save the dataset
# ------------------------------------------

df.to_csv("../data/phase1_data.csv", index=False)

print("\nPhase 1 dataset saved!")