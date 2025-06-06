import math
from data import ASTEROID_MAX_SPEED, HEIGHT, WIDTH
from game import (
    Asteroid,
    AsteroidInfo,
    Mineral,
    Spaceship,
    count_asteroids_in_radius,
    get_closest_asteroid_info,
    get_safest_minerals,
)

# TOTAL INPUTS: 4 (ship state) + 8 (asteroid info) + 4 (mineral info) = 16 inputs
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

    asteroid_count_around_ship = len(count_asteroids_in_radius(ship, asteroids, 100))

    # Ship State (4 inputs_value)
    inputs_value.extend(
        [
            ship.fuel / 100.0,  # Normalize fuel (0 to 1)
            math.sin(ship.angle),  # Ship heading Y component
            math.cos(ship.angle),  # Ship heading X component
            asteroid_count_around_ship / 10.0,  # Normalize asteroid count (0 to 1)
        ]
    )
    inputs_explanation.extend(
        [
            "Ship Fuel (normalized)",
            "Ship Heading Sin (normalized)",
            "Ship Heading Cos (normalized)",
            "Asteroid Count (normalized)",
        ]
    )
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Ship State"
    )

    MAX_DISTANCE = math.sqrt(WIDTH**2 + HEIGHT**2) / 2

    # Enhanced Asteroid Info (8 inputs per asteroid)
    closest_asteroids = get_closest_asteroid_info(ship, asteroids, top_n=1)
    for i in range(1):
        if i < len(closest_asteroids):
            asteroid: AsteroidInfo = closest_asteroids[i]

            # 1. Current position and distance
            current_distance = 1.0 - 2.0 * (asteroid.distance / MAX_DISTANCE)  # [-1, 1]
            current_angle_sin = math.sin(asteroid.relative_angle)
            current_angle_cos = math.cos(asteroid.relative_angle)

            # 2. Velocity information - CRITICAL for avoidance!
            velocity_direction_sin = math.sin(
                asteroid.velocity_angle
            )  # Direction it's moving
            velocity_direction_cos = math.cos(asteroid.velocity_angle)
            velocity_magnitude = min(
                1.0, asteroid.velocity_magnitude / ASTEROID_MAX_SPEED
            )  # [0, 1]

            # 3. Threat level - Is asteroid approaching us?
            # Calculate if asteroid is moving toward ship
            approach_threat = 0.0
            if asteroid.velocity_magnitude > 0:
                # Vector from asteroid to ship
                ship_direction_sin = math.sin(
                    asteroid.relative_angle + math.pi
                )  # Opposite direction
                ship_direction_cos = math.cos(asteroid.relative_angle + math.pi)

                # Dot product: positive if asteroid is moving toward ship
                approach_dot = (
                    velocity_direction_sin * ship_direction_sin
                    + velocity_direction_cos * ship_direction_cos
                )
                approach_threat = max(
                    0.0, approach_dot
                )  # [0, 1] where 1 = direct approach

            # 4. Future collision risk - Will it hit us soon?
            future_collision_risk = 0.0
            if len(asteroid.future_positions) > 0:
                # Check closest future position (30 frames ahead)
                future_distance, _ = asteroid.future_positions[0]
                if future_distance < 50:  # Danger threshold
                    future_collision_risk = 1.0 - (future_distance / 50.0)  # [0, 1]

            inputs_value.extend(
                [
                    current_distance,  # How close is it now?
                    current_angle_sin,  # Where is it relative to us?
                    current_angle_cos,
                    velocity_direction_sin,  # Which way is it moving?
                    velocity_direction_cos,
                    velocity_magnitude,  # How fast is it moving?
                    approach_threat,  # Is it coming toward us?
                    future_collision_risk,  # Will it hit us soon?
                ]
            )
        else:
            # No asteroid - safe defaults
            inputs_value.extend(
                [
                    -1.0,  # Far away
                    0.0,
                    1.0,  # Straight ahead (safe default)
                    0.0,
                    1.0,  # Not moving toward us
                    0.0,  # No velocity
                    0.0,  # No approach threat
                    0.0,  # No collision risk
                ]
            )

        inputs_explanation.extend(
            [
                f"Asteroid {i + 1} Distance (-1=far, 1=close)",
                f"Asteroid {i + 1} Relative Angle Sin (-1 to 1)",
                f"Asteroid {i + 1} Relative Angle Cos (-1 to 1)",
                f"Asteroid {i + 1} Velocity Direction Sin (-1 to 1)",
                f"Asteroid {i + 1} Velocity Direction Cos (-1 to 1)",
                f"Asteroid {i + 1} Velocity Magnitude (0=still, 1=max speed)",
                f"Asteroid {i + 1} Approach Threat (0=safe, 1=direct approach)",
                f"Asteroid {i + 1} Future Collision Risk (0=safe, 1=imminent)",
            ]
        )
    safest_minerals = get_safest_minerals(ship, minerals, asteroids, top_n=1)

    for i in range(1):
        if i < len(safest_minerals):
            mineral_info = safest_minerals[i]
            inputs_value.extend(
                [
                    max(0.0, 1.0 - (mineral_info.distance / MAX_DISTANCE)),
                    math.sin(mineral_info.relative_angle),  # [-1, 1] range
                    math.cos(mineral_info.relative_angle),  # [-1, 1] range
                    mineral_info.normalized_risk,  # Already properly normalized [0,1]
                ]
            )
        else:
            inputs_value.extend([0.0, 0.0, 1.0, 0.0])  # No mineral available

        inputs_explanation.extend(
            [
                f"Safe Mineral {i + 1} Distance (normalized)",
                f"Safe Mineral {i + 1} Relative Angle Sin (-1 to 1)",
                f"Safe Mineral {i + 1} Relative Angle Cos (-1 to 1)",
                f"Safe Mineral {i + 1} Safety Score (0=dangerous, 1=safe)",
            ]
        )

    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Mineral Info"
    )

    return inputs_explanation, inputs_value
