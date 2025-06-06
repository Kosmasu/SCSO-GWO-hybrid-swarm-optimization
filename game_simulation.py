import math
import time
import pygame
from typing import Tuple, List, Callable
from game import Asteroid, Mineral, Spaceship
from data import BLACK


class SimulationConfig:
    """Configuration for game simulation"""

    def __init__(self):
        self.num_minerals = 5
        self.num_asteroids = 10
        self.min_asteroid_distance = 100
        self.max_timeout_frames = 100_000
        self.visualization_fps = 60


class SimulationResult:
    """Result of a simulation run"""

    def __init__(self):
        self.death_reason = "unknown"
        self.alive_frames = 0
        self.total_fuel_gain: float = 0.0
        self.minerals_collected: int = 0
        self.final_fitness: float = 0.0
        self.backward_movement_ratio: float = 0.0
        self.spinning_penalty: float = 0.0


def handle_steering(output: List[float]) -> float:
    """Handle steering based on neural network output"""
    turn_output = output[0]
    if turn_output < -0.3:
        return ((turn_output + 0.3) / 0.7) * 0.15  # Full left
    elif turn_output >= -0.3 and turn_output <= 0.3:
        return 0  # No turn
    else:
        return ((turn_output - 0.3) / 0.7) * 0.15  # Full right


def handle_thrusting(output: List[float]) -> float:
    """Handle thrusting based on neural network output"""
    thrust_output = output[1]
    if thrust_output < -0.3:
        return ((thrust_output + 0.3) / 0.7) * 0.8  # Full backward
    elif thrust_output >= -0.3 and thrust_output <= 0.3:
        return 0  # No thrust
    else:
        return (thrust_output - 0.3) / 0.7  # Full forward


def initialize_game_objects(
    config: SimulationConfig,
) -> Tuple[Spaceship, List[Mineral], List[Asteroid]]:
    """Initialize ship, minerals, and asteroids"""
    ship = Spaceship()

    # Initialize minerals
    minerals = [Mineral() for _ in range(config.num_minerals)]

    # Initialize asteroids (ensure they're not too close to ship)
    asteroids = []
    for _ in range(config.num_asteroids):
        asteroid = Asteroid()
        while (
            math.hypot(ship.x - asteroid.x, ship.y - asteroid.y)
            < ship.radius + asteroid.radius + config.min_asteroid_distance
        ):
            asteroid = Asteroid()
        asteroids.append(asteroid)

    return ship, minerals, asteroids


def update_game_state(
    ship: Spaceship,
    minerals: List[Mineral],
    asteroids: List[Asteroid],
    thrust_power: float,
    config: SimulationConfig,
) -> float:
    """Update game state and return fuel gain"""
    # Move ship
    dx = thrust_power * ship.speed * math.cos(ship.angle)
    dy = thrust_power * ship.speed * math.sin(ship.angle)
    ship.move(dx, dy)

    # Handle mining
    fuel_before = ship.fuel
    ship.mine(minerals)
    if len(minerals) < config.num_minerals:
        minerals.append(Mineral())
    fuel_gain = ship.fuel - fuel_before

    # Move asteroids
    for asteroid in asteroids:
        asteroid.move()

    return fuel_gain if fuel_gain > 0 else 0


def check_termination_conditions(
    ship: Spaceship,
    asteroids: List[Asteroid],
    alive_frames: int,
    config: SimulationConfig,
) -> Tuple[bool, str]:
    """Check if simulation should terminate and return reason"""
    if asteroids:
        closest_asteroid = min(
            (a for a in asteroids), key=lambda a: math.hypot(ship.x - a.x, ship.y - a.y)
        )
        asteroid_collision = (
            math.hypot(ship.x - closest_asteroid.x, ship.y - closest_asteroid.y)
            < ship.radius + closest_asteroid.radius
        )
        if asteroid_collision:
            return True, "asteroid_collision"

    if ship.fuel <= 0:
        return True, "out_of_fuel"

    if alive_frames >= config.max_timeout_frames:
        return True, "timeout"

    return False, "alive"


def calculate_fitness(
    ship: Spaceship,
    alive_frames: int,
    total_fuel_gain: float,
    backward_penalty: float,
    spinning_penalty: float,
    stillness_bonus: float = 0.0,
) -> float:
    """Calculate fitness score"""
    return (
        (alive_frames / 4)
        + total_fuel_gain
        + ship.minerals * 20
        + ship.fuel * 0.5
        + stillness_bonus
        - backward_penalty
        - spinning_penalty
    )


def run_neat_simulation(
    network,
    get_inputs_func: Callable,
    config: SimulationConfig,
    visualizer=None,
    screen=None,
    clock=None,
) -> SimulationResult:
    """
    Run a complete NEAT simulation

    Args:
        network: NEAT neural network
        get_inputs_func: Function to get NEAT inputs (ship, minerals, asteroids) -> (explanation, values)
        config: Simulation configuration
        visualizer: Optional visualizer for display
        screen: Pygame screen for visualization
        clock: Pygame clock for visualization
    """
    result = SimulationResult()

    # Initialize game objects
    ship, minerals, asteroids = initialize_game_objects(config)

    # Tracking variables
    alive_frame_counter = 0
    total_fuel_gain = 0
    backward_movement_counter = 0
    total_movement_counter = 0
    total_angle_change = 0
    still_frames = 0  # Add stillness tracking

    if visualizer:
        visualizer.start_time = time.time()

    while True:
        alive_frame_counter += 1

        # Handle pygame events if visualizing
        if visualizer and screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return result

        # Get neural network inputs
        _, inputs_value = get_inputs_func(ship, minerals, asteroids)

        # Get network output
        output = network.activate(inputs_value)

        # Track angle changes for spinning penalty
        old_angle = ship.angle
        ship.angle += handle_steering(output)
        ship.angle = ship.angle % (2 * math.pi)

        # Calculate angle change (handle wrapping)
        angle_diff = abs(ship.angle - old_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        total_angle_change += angle_diff

        # Handle thrusting and track movement
        thrust_power = handle_thrusting(output)

        if thrust_power == 0:
            still_frames += 1
        else:
            total_movement_counter += 1
            if thrust_power < 0:
                backward_movement_counter += 1

        # Update game state
        fuel_gain = update_game_state(ship, minerals, asteroids, thrust_power, config)
        total_fuel_gain += fuel_gain

        # Calculate penalties
        backward_penalty = 0
        if total_movement_counter > 50:
            backward_ratio = backward_movement_counter / total_movement_counter
            if backward_ratio > 0.5:
                backward_penalty = (backward_ratio - 0.5) * 200

        spinning_penalty = 0
        if alive_frame_counter > 100:
            avg_angle_change_per_frame = total_angle_change / alive_frame_counter
            if avg_angle_change_per_frame > 0.10:
                spinning_penalty = (avg_angle_change_per_frame - 0.05) * 1000

        # Calculate stillness bonus
        stillness_ratio = (
            still_frames / alive_frame_counter if alive_frame_counter > 0 else 0
        )
        stillness_bonus = stillness_ratio * 50  # Adjust multiplier as needed

        # Calculate current fitness
        current_fitness = calculate_fitness(
            ship,
            alive_frame_counter,
            total_fuel_gain,
            backward_penalty,
            spinning_penalty,
            stillness_bonus,
        )
        # Visualization
        if visualizer and screen and clock:
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw(screen)
            for asteroid in asteroids:
                asteroid.draw(screen)
            ship.draw(screen)

            visualizer.draw_stats(screen, current_fitness, ship.minerals, ship.fuel)
            pygame.display.flip()
            clock.tick(config.visualization_fps)

        # Check termination conditions
        should_terminate, death_reason = check_termination_conditions(
            ship, asteroids, alive_frame_counter, config
        )

        if should_terminate:
            result.death_reason = death_reason
            result.alive_frames = alive_frame_counter
            result.total_fuel_gain = total_fuel_gain
            result.minerals_collected = ship.minerals
            result.backward_movement_ratio = (
                backward_movement_counter / total_movement_counter
                if total_movement_counter > 0
                else 0
            )
            result.spinning_penalty = spinning_penalty

            result.final_fitness = current_fitness
            break

    return result


def run_manual_simulation(
    ship: Spaceship,
    minerals: List[Mineral],
    asteroids: List[Asteroid],
    control_func: Callable[[List], Tuple[float, float]],
    config: SimulationConfig,
) -> SimulationResult:
    """
    Run simulation with manual control function

    Args:
        ship: Spaceship instance
        minerals: List of minerals
        asteroids: List of asteroids
        control_func: Function that takes (ship, minerals, asteroids) and returns (turn_rate, thrust_power)
        config: Simulation configuration
    """
    result = SimulationResult()
    alive_frame_counter = 0
    total_fuel_gain = 0

    while True:
        alive_frame_counter += 1

        # Get control inputs
        turn_rate, thrust_power = control_func([ship, minerals, asteroids])

        # Apply controls
        ship.angle += turn_rate
        ship.angle = ship.angle % (2 * math.pi)

        # Update game state
        fuel_gain = update_game_state(ship, minerals, asteroids, thrust_power, config)
        total_fuel_gain += fuel_gain

        # Check termination
        should_terminate, death_reason = check_termination_conditions(
            ship, asteroids, alive_frame_counter, config
        )

        if should_terminate:
            result.death_reason = death_reason
            result.alive_frames = alive_frame_counter
            result.total_fuel_gain = total_fuel_gain
            result.minerals_collected = ship.minerals
            result.final_fitness = calculate_fitness(
                alive_frame_counter, total_fuel_gain, 0, 0
            )
            break

    return result
