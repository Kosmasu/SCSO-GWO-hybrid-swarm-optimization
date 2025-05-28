import pickle
import pygame
import math
import os
import neat
import time

from custom_reporter import DataReporter
from data import BLACK, ASTEROID_MAX_RADIUS, WIDTH, HEIGHT, WHITE
from game import (
    Asteroid,
    Mineral,
    Spaceship,
    count_asteroids_in_radius,
    get_closest_asteroid_info,
    get_closest_mineral_info,
)

if not pygame.get_init():
    pygame.init()

def get_neat_inputs(
    ship: Spaceship, minerals: list[Mineral], asteroids: list[Asteroid]
) -> tuple[list[str], list[float]]:
    """
    Generate normalized inputs_value for NEAT neural network.

    Returns:
        list[float]: Normalized input values for the neural network
    """
    # Inputs explanation and values
    # Angles:
    #   math.sin(angle) and math.cos(angle) are used to represent angles in a normalized way
    #   where angle is in radians, and the range is [-1, 1].

    inputs_explanation: list[str] = []
    inputs_value: list[float] = []

    # Proximity Asteroids
    inputs_value.extend([
        1.0 if len(count_asteroids_in_radius(ship, asteroids, radius=50)) > 0 else 0.0,    # Any immediate danger
        1.0 if len(count_asteroids_in_radius(ship, asteroids, radius=50)) > 1 else 0.0,    # Multiple immediate threats
        1.0 if len(count_asteroids_in_radius(ship, asteroids, radius=100)) > 2 else 0.0,   # Crowded area
    ])
    inputs_explanation.extend([
        "Immediate Danger (0 or 1): Any asteroid within 50 pixels",
        "Multiple Threats (0 or 1): More than one asteroid within 50 pixels",
        "Crowded Area (0 or 1): More than two asteroids within 100 pixels",
    ])
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Proximity Asteroids"
    )

    max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)

    # Top 5 Asteroid Information
    asteroid_info = get_closest_asteroid_info(ship, asteroids, top_n=5)
    for i in range(5):
        if i < len(asteroid_info):
            info = asteroid_info[i]
            inputs_value.extend([
                min(info.distance / max_distance, 1.0),  # Distance (normalized)
                math.sin(info.relative_angle),  # Relative angle Y (normalized to -1, 1)
                math.cos(info.relative_angle),  # Relative angle X (normalized to -1, 1)
            ])

            # Add trend: is asteroid getting closer or farther?
            if len(info.future_positions) >= 2:
                trend = (
                    info.future_positions[-1][0] - info.future_positions[0][0]
                )  # Last - First distance
                inputs_value.append(
                    max(-1.0, min(1.0, trend / max_distance))
                )  # Normalize trend (-1 = approaching, +1 = moving away)
            else:
                inputs_value.append(0.0)  # No trend data
            inputs_value.append(
                info.asteroid.radius / ASTEROID_MAX_RADIUS
            )  # Asteroid size normalized

            # Asteroid speed magnitude (normalized)
            speed_magnitude = math.sqrt(
                info.asteroid.speed_x**2 + info.asteroid.speed_y**2
            )
            max_speed_magnitude = math.sqrt(2 * (2.0**2))  # sqrt(8) ≈ 2.83
            inputs_value.append(
                min(speed_magnitude / max_speed_magnitude, 1.0)
            )  # Normalize by actual max speed

            # Asteroid speed direction (using sin/cos like angles)
            if speed_magnitude > 0:
                speed_angle = math.atan2(info.asteroid.speed_y, info.asteroid.speed_x)
                inputs_value.extend([
                    math.sin(speed_angle),  # Speed direction Y component
                    math.cos(speed_angle),  # Speed direction X component
                ])
            else:
                inputs_value.extend([0.0, 0.0])  # No movement
        else:
            # No asteroid data - use -1 to indicate non-existent asteroid
            inputs_value.extend(
                [
                    1.0,  # Distance: maximum distance = very far away
                    0.0,  # Relative angle sin: neutral direction
                    0.0,  # Relative angle cos: neutral direction
                    0.0,  # Trend: no change
                    0.0,  # Radius: minimum size
                    0.0,  # Speed magnitude: not moving
                    0.0,  # Speed direction sin: not moving
                    0.0,  # Speed direction cos: not moving
                ]
            )
        inputs_explanation.extend(
            [
                f"Asteroid {i + 1} Distance (normalized)",
                f"Asteroid {i + 1} Rel Angle Sin (normalized)",
                f"Asteroid {i + 1} Rel Angle Cos (normalized)",
                f"Asteroid {i + 1} Trend (normalized)",
                f"Asteroid {i + 1} Radius (normalized)",
                f"Asteroid {i + 1} Speed Mag (normalized)",
                f"Asteroid {i + 1} Speed Dir Sin (normalized)",
                f"Asteroid {i + 1} Speed Dir Cos (normalized)",
            ]
        )
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Asteroid Information"
    )

    # Top 3 Mineral Information
    mineral_info = get_closest_mineral_info(ship, minerals, top_n=3)
    for i in range(3):
        if i < len(mineral_info):
            info = mineral_info[i]

            inputs_value.append(
                min(info.distance / max_distance, 1.0)
            )  # Distance (normalized)

            inputs_value.append(
                math.sin(info.relative_angle)
            )  # Relative angle (Y component)
            inputs_value.append(
                math.cos(info.relative_angle)
            )  # Relative angle (X component)

            # Reward factor based on distance to mineral
            reward_factor = 1.0 - (info.distance / max_distance)
            inputs_value.append(
                reward_factor
            )  # Base reward (inverse distance - closer = better)

            # Asteroid density around mineral (risk factor)
            inputs_value.append(
                len(count_asteroids_in_radius(info.mineral, asteroids, radius=50))
                / len(asteroids)
            )  # Immediate danger
            inputs_value.append(
                len(count_asteroids_in_radius(info.mineral, asteroids, radius=100))
                / len(asteroids)
            )  # Medium danger
            inputs_value.append(
                len(count_asteroids_in_radius(info.mineral, asteroids, radius=150))
                / len(asteroids)
            )  # Far danger
        else:
            # No mineral data - fill with safe neutral values
            inputs_value.extend(
                [
                    1.0,  # Distance: maximum distance = very far away
                    0.0,  # Relative angle sin: neutral direction
                    0.0,  # Relative angle cos: neutral direction
                    0.0,  # Reward factor: no reward
                    0.0,  # Immediate danger: no danger
                    0.0,  # Medium danger: no danger
                    0.0,  # Far danger: no danger
                ]
            )
        inputs_explanation.extend(
            [
                f"Mineral {i + 1} Distance (normalized)",
                f"Mineral {i + 1} Rel Angle Sin (normalized)",
                f"Mineral {i + 1} Rel Angle Cos (normalized)",
                f"Mineral {i + 1} Reward (normalized)",
                f"Mineral {i + 1} Immediate Danger (normalized)",
                f"Mineral {i + 1} Medium Danger (normalized)",
                f"Mineral {i + 1} Far Danger (normalized)",
            ]
        )
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Mineral Information"
    )

    # Ship State (5 inputs_value)
    is_moving = 1.0 if (ship.velocity_x != 0 or ship.velocity_y != 0) else 0.0

    inputs_value.extend(
        [
            ship.fuel / 100.0,  # Normalize fuel (0 to 1)
            is_moving,  # Binary: is ship currently moving? (0 or 1)
            ship.minerals,  # Total minerals collected
            math.sin(ship.angle),  # Ship heading Y component
            math.cos(ship.angle),  # Ship heading X component
        ]
    )
    inputs_explanation.extend(
        [
            "Ship Fuel (normalized)",
            "Ship Is Moving (0 or 1)",
            "Ship Minerals Collected",
            "Ship Heading Sin (normalized)",
            "Ship Heading Cos (normalized)",
        ]
    )
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Ship State"
    )

    # Spatial Awareness (4 inputs_value)
    # Distance to edges (for wrapping awareness)
    inputs_value.extend(
        [
            ship.x / WIDTH,  # Distance to left edge (normalized)
            (WIDTH - ship.x) / WIDTH,  # Distance to right edge (normalized)
            ship.y / HEIGHT,  # Distance to top edge (normalized)
            (HEIGHT - ship.y) / HEIGHT,  # Distance to bottom edge (normalized)
        ]
    )
    inputs_explanation.extend(
        [
            "Ship Distance to Left Edge (normalized)",
            "Ship Distance to Right Edge (normalized)",
            "Ship Distance to Top Edge (normalized)",
            "Ship Distance to Bottom Edge (normalized)",
        ]
    )
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Spatial Awareness"
    )
    return inputs_explanation, inputs_value


def run_simulation(genome, config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship()
    minerals: list[Mineral] = [Mineral() for _ in range(5)]
    asteroids: list[Asteroid] = []

    # Generate initial asteroids
    for _ in range(8):
        asteroid = Asteroid()
        # Ensure asteroids are not too close to the ship. At least 100 pixels away
        while (
            math.hypot(ship.x - asteroid.x, ship.y - asteroid.y)
            < ship.radius + asteroid.radius + 100
        ):
            asteroid = Asteroid()
        asteroids.append(asteroid)

    alive_frame_counter = 0
    dx, dy = 0, 0

    if visualizer:
        visualizer.start_time = time.time()

    while True:
        alive_frame_counter += 1

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        _, inputs_value = get_neat_inputs(ship, minerals, asteroids)

        # Get actions from network
        output = net.activate(inputs_value)

        # Execute actions with improved mapping
        ship.angle += (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
        ship.angle = ship.angle % (2 * math.pi)  # Normalize angle

        dx, dy = 0, 0
        if output[1] > 0.5:  # Thrust (lowered threshold)
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)

        if output[2] > 0.5:
            ship.mine(minerals)
            if len(minerals) < 3:
                minerals.extend(Mineral() for _ in range(2))

        # Move asteroids
        for asteroid in asteroids:
            asteroid.move()

        # Enhanced fitness function
        # 1. Survival time with linear growth
        survival_bonus = alive_frame_counter * 0.1  # Linear growth

        # 2. Fuel level (reward for not wasting fuel)
        # Fuel is capped at 100, so we can use it directly
        fuel_level = ship.fuel

        # Combine fitness components
        genome.fitness = (
            survival_bonus  # Main objective
            + fuel_level  # Don't waste fuel
        )

        # Visualization
        if visualizer:
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw(screen)

            for asteroid in asteroids:
                asteroid.draw(screen)

            ship.draw(screen)

            visualizer.draw_stats(screen, genome.fitness, ship.minerals, ship.fuel)
            pygame.display.flip()
            clock.tick(30)

        closest_asteroid = min(
            (a for a in asteroids), key=lambda a: math.hypot(ship.x - a.x, ship.y - a.y)
        )
        # Termination conditions
        asteroid_collision = (
            math.hypot(ship.x - closest_asteroid.x, ship.y - closest_asteroid.y)
            < ship.radius + closest_asteroid.radius
        )
        out_of_fuel = ship.fuel <= 0

        # Adaptive timeout based on performance and generation
        generation_bonus = min(20_000, config.visualizer.generation * 50)  # More time for later generations
        base_timeout_frame = 12_000 + generation_bonus
        
        # Mineral-based bonus (encourages mineral collection)
        mineral_bonus = min(18_000, ship.minerals * 1000)
            
        max_timeout_frame = base_timeout_frame + mineral_bonus
        # Cap at reasonable maximum
        max_timeout_frame = min(max_timeout_frame, 40_000)

        if asteroid_collision or out_of_fuel or alive_frame_counter >= max_timeout_frame:
            break


FONT = pygame.font.SysFont(None, 36)


class TrainingVisualizer:
    def __init__(self):
        self.best_fitness = -float("inf")
        self.generation = 0
        self.start_time = time.time()

    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")
        print(f"Generation {self.generation} best: {best_genome.fitness:.1f}")

    def draw_stats(self, screen, fitness, minerals, fuel):
        stats = [
            f"Gen: {self.generation}",
            f"Alive Time: {int(time.time() - self.start_time)}s",
            f"Fitness: {fitness:.1f}",
            f"Best: {self.best_fitness:.1f}",
            f"Minerals: {minerals}",
            f"Fuel: {fuel:.1f}",
        ]

        for i, stat in enumerate(stats):
            text = FONT.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 40))


def eval_genomes(genomes, config):
    visualizer = config.visualizer

    # First evaluate all genomes to find the best
    best_in_generation = None
    best_fitness = -float("inf")

    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        run_simulation(
            genome, config, visualizer=None
        )  # No visualization during evaluation

        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome

    # Update visualizer with this generation's results
    visualizer.update_generation(best_in_generation)

    # Visualize the best genome from this generation (every 5th generation)
    if best_in_generation and visualizer.generation % 5 == 0:
        print(
            f"Displaying generation {visualizer.generation} best (Fitness: {best_fitness:.1f})"
        )
        run_simulation(
            best_in_generation, config, visualizer=visualizer
        )  # With visualization


def run_neat(config_file: str, output_dir: str, continue_from_checkpoint: bool = False):
    # Initialize pygame
    global screen, clock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()

    # Create and store visualizer in config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    config.visualizer = TrainingVisualizer()

    # Create population
    if continue_from_checkpoint:
        print("Restoring from checkpoint...")
        # open `info_log.json` to get the last generation
        info_log = os.path.join(output_dir, "info_log.json")
        if not os.path.exists(info_log):
            raise FileNotFoundError(
                f"Checkpoint info log not found: {info_log}. "
                "Make sure to run the training first."
            )
        with open(info_log, "r") as f:
            import json

            info = json.load(f)
            last_generation = info[-1]["generation"]
            print(f"Last generation: {last_generation}")
        population = neat.Checkpointer.restore_checkpoint(
            os.path.join(output_dir, "checkpoints", f"{last_generation}")
        )
        print(f"Restored population from generation {population.generation}")
        # Sync visualizer with the restored population
        config.visualizer.generation = population.generation
    else:
        population = neat.Population(config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(25, filename_prefix=os.path.join(output_dir, "checkpoints/"))
    )
    population.add_reporter(DataReporter(output_dir=output_dir))
    # Run NEAT
    try:
        winner = population.run(eval_genomes, 1000)

        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")

        # Test the winner
        print("\nTesting winner...")
        run_simulation(winner, config, visualizer=config.visualizer)

        # save the winner
        with open(os.path.join(output_dir, "winner.pkl"), "wb") as f:
            pickle.dump(winner, f)
        print(f"Winner genome saved to {output_dir} as 'winner.pkl'")

    finally:
        pygame.quit()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/neat/{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "checkpoints/", exist_ok=True)
    run_neat(config_file, output_dir)
