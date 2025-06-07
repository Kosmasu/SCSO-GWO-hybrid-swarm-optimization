import math
from data import ASTEROID_MAX_SPEED, HEIGHT, WIDTH
from game import (
    Asteroid,
    Mineral,
    Spaceship,
    get_closest_asteroid_info,
    get_closest_mineral_info,
)


# TOTAL INPUTS: 3 (ship state) + 2 (asteroid info) + 2 (mineral info) = 7 inputs
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

    MAX_DISTANCE = math.sqrt(WIDTH**2 + HEIGHT**2)

    # Ship State (3 inputs)
    inputs_value.extend(
        [
            ship.fuel / 100.0,  # Normalize fuel (0 to 1)
            ship.angle / math.pi,  # Normalize angle to [-1, 1]
        ]
    )
    inputs_explanation.extend(
        [
            "Ship Fuel (normalized)",
            "Ship Heading (normalized)",
        ]
    )

    # Closest Asteroid (2 inputs)
    closest_asteroids = get_closest_asteroid_info(ship, asteroids, top_n=1)
    if closest_asteroids:
        asteroid = closest_asteroids[0]
        inputs_value.extend(
            [
                1.0 - (asteroid.distance / MAX_DISTANCE),  # Closer = higher value
                asteroid.relative_angle / math.pi,  # Normalize angle to [-1, 1]
            ]
        )
    else:
        inputs_value.extend([0.0, 0.0])  # No asteroid nearby

    inputs_explanation.extend(
        [
            "Closest Asteroid Distance (1=close, 0=far)",
            "Closest Asteroid Angle (normalized)",
        ]
    )

    # Closest Mineral (2 inputs)
    closest_minerals = get_closest_mineral_info(ship, minerals, top_n=1)
    if closest_minerals:
        mineral = closest_minerals[0]
        inputs_value.extend(
            [
                1.0 - (mineral.distance / MAX_DISTANCE),  # Closer = higher value
                mineral.relative_angle / math.pi,  # Normalize angle to [-1, 1]
            ]
        )
    else:
        inputs_value.extend([0.0, 0.0])  # No mineral nearby

    inputs_explanation.extend(
        [
            "Closest Mineral Distance (1=close, 0=far)",
            "Closest Mineral Angle (normalized)",
        ]
    )

    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length"
    )

    return inputs_explanation, inputs_value
