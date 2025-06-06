
import math
from game import Asteroid, Mineral, Spaceship, get_closest_mineral_info, radar_scan

# TOTAL INPUTS: 3 (ship state) + 12 (radar scan) + 9 (minerals) = 24 inputs
def get_neat_inputs(
    ship: Spaceship, minerals: list[Mineral], asteroids: list[Asteroid]
) -> tuple[list[str], list[float]]:
    """
    Generate normalized inputs_value for NEAT neural network.

    Returns:
        list[float]: Normalized input values for the neural network
    """

    inputs_explanation: list[str] = []
    inputs_value: list[float] = []

    # Ship State (3 inputs_value)
    inputs_value.extend(
        [
            ship.fuel / 100.0,  # Normalize fuel (0 to 1)
            math.sin(ship.angle),  # Ship heading Y component
            math.cos(ship.angle),  # Ship heading X component
        ]
    )
    inputs_explanation.extend(
        [
            "Ship Fuel (normalized)",
            "Ship Heading Sin (normalized)",
            "Ship Heading Cos (normalized)",
        ]
    )
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Ship State"
    )

    N_RADAR_DIRECTIONS = 12
    MAX_RADAR_RANGE = 200.0
    # Radar Scan Asteroids (12 inputs_value)
    radar_scan_results = radar_scan(
        ship, asteroids, n_directions=N_RADAR_DIRECTIONS, max_range=MAX_RADAR_RANGE
    )

    # Normalize radar distances (0 = max range/no obstacle, 1 = touching/collision)
    normalized_radar = [
        1.0 - (result.distance / MAX_RADAR_RANGE) for result in radar_scan_results
    ]

    inputs_value.extend(normalized_radar)

    # Generate explanations for each radar direction (relative to ship heading)
    for i in range(N_RADAR_DIRECTIONS):
        angle_deg = i * (360 / N_RADAR_DIRECTIONS)
        inputs_explanation.append(
            f"Asteroid Radar {angle_deg:.0f}° relative (normalized inverse distance)"
        )

    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Asteroid Radar Scan"
    )

    # Top 3 Closest Minerals (9 inputs_value)
    closest_minerals = get_closest_mineral_info(ship, minerals, top_n=3)

    for i in range(3):
        if i < len(closest_minerals):
            mineral = closest_minerals[i]
            inputs_value.extend(
                [
                    max(
                        0.0, 1.0 - (mineral.distance / MAX_RADAR_RANGE)
                    ),  # Normalize distance
                    math.sin(mineral.relative_angle),  # Y component of angle
                    math.cos(mineral.relative_angle),  # X component of angle
                ]
            )
        else:
            # Pad with safe default values for missing minerals
            # Use max distance (1.0) and neutral angles (0.0, 1.0) for cos(0)
            inputs_value.extend([1.0, 0.0, 1.0])

        inputs_explanation.extend(
            [
                f"Mineral {i + 1} Distance (normalized)",
                f"Mineral {i + 1} Relative Angle Sin (normalized)",
                f"Mineral {i + 1} Relative Angle Cos (normalized)",
            ]
        )

    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Mineral Info"
    )

    return inputs_explanation, inputs_value
