"""
Simulation & Model Configuration
Defines constants, hyperparameters, and scoring weights for the traffic intelligence system.
"""

# City Dimensions
NUM_ZONES = 20
NUM_WEEKS = 24
RANDOM_SEED = 42

# Zone baseline bounds
POPULATION_MIN = 1000
POPULATION_MAX = 15000
ROAD_CAPACITY_MIN = 40
ROAD_CAPACITY_MAX = 150

# Weather definitions (Multiplier impact on speed and congestion)
WEATHER_PROBABILITIES = {
    "Clear": 0.60,
    "Rain": 0.25,
    "Heavy_Rain": 0.10,
    "Fog": 0.05,
}

WEATHER_IMPACT = {
    "Clear": {"congestion_mult": 1.00, "speed_mult": 1.00, "risk_bias": 0.00},
    "Rain": {"congestion_mult": 1.15, "speed_mult": 0.85, "risk_bias": 0.12},
    "Heavy_Rain": {"congestion_mult": 1.35, "speed_mult": 0.70, "risk_bias": 0.28},
    "Fog": {"congestion_mult": 1.20, "speed_mult": 0.75, "risk_bias": 0.20},
}

# Special Events definitions
EVENT_PROBABILITIES = {
    "None": 0.85,
    "Festival": 0.08,
    "Roadwork": 0.07,
}

EVENT_IMPACT = {
    "None": {"veh_mult": 1.0, "cap_mult": 1.0, "risk_bias": 0.0},
    "Festival": {"veh_mult": 1.45, "cap_mult": 0.9, "risk_bias": 0.18},
    "Roadwork": {"veh_mult": 1.0, "cap_mult": 0.65, "risk_bias": 0.22},
}

# Exposure Formula Weights (Sums to 1.0)
WEIGHT_EXPOSURE_POP = 0.55
WEIGHT_EXPOSURE_VEH = 0.45

# Priority Formula Weights (Sums to 1.0)
WEIGHT_PRIORITY_RISK = 0.50
WEIGHT_PRIORITY_EXPOSURE = 0.35
WEIGHT_PRIORITY_TREND = 0.15

# Priority Tier Thresholds
PRIORITY_TIERS = [
    (80, "CRITICAL", "[CRITICAL]"),
    (65, "VERY HIGH", "[VERY HIGH]"),
    (50, "HIGH", "[HIGH]"),
    (0, "MODERATE", "[MODERATE]"),
]
