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
    inputs_explanation.append("Proximity Asteroids - Radius 50")
    inputs_value.append(
        len(count_asteroids_in_radius(ship, asteroids, radius=50)) / len(asteroids)
    )  # Immediate danger
    inputs_explanation.append("Proximity Asteroids - Radius 100")
    inputs_value.append(
        len(count_asteroids_in_radius(ship, asteroids, radius=100)) / len(asteroids)
    )  # Medium danger
    inputs_explanation.append("Proximity Asteroids - Radius 150")
    inputs_value.append(
        len(count_asteroids_in_radius(ship, asteroids, radius=150)) / len(asteroids)
    )  # Far danger
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Proximity Asteroids"
    )

    max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)

    # Top 5 Asteroid Information
    asteroid_info = get_closest_asteroid_info(ship, asteroids, top_n=5)
    for i in range(5):
        if i < len(asteroid_info):
            info = asteroid_info[i]
            inputs_value.append(
                min(info.distance / max_distance, 1.0)
            )  # Current distance (normalized)
            inputs_value.append(
                math.sin(info.relative_angle)
            )  # Relative angle Y (normalized to -1, 1)
            inputs_value.append(
                math.cos(info.relative_angle)
            )  # Relative angle X (normalized to -1, 1)

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
                inputs_value.append(
                    math.sin(speed_angle)
                )  # Speed direction Y component
                inputs_value.append(
                    math.cos(speed_angle)
                )  # Speed direction X component
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

    # Top 5 Mineral Information
    mineral_info = get_closest_mineral_info(ship, minerals, top_n=5)
    for i in range(5):
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
    inputs_value.extend(
        [
            ship.fuel / 100.0,  # Normalize fuel (0 to 1)
            ship.speed / 10.0,  # Normalize speed (0 to 10)
            ship.minerals,  # Total minerals collected
            math.sin(ship.angle),  # Ship heading Y component
            math.cos(ship.angle),  # Ship heading X component
        ]
    )
    inputs_explanation.extend(
        [
            "Ship Fuel (normalized)",
            "Ship Speed (normalized)",
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

    alive_time = 0
    previous_mineral_distance = float("inf")
    movement_towards_minerals = 0

    while True:
        alive_time += 1

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

        # Calculate movement towards minerals reward
        if minerals:  # Avoid first frame issues
            # Find closest mineral
            closest_mineral = min(
                minerals, key=lambda m: math.hypot(ship.x - m.x, ship.y - m.y)
            )
            current_mineral_distance = math.hypot(
                ship.x - closest_mineral.x, ship.y - closest_mineral.y
            )

            # Reward for moving closer to minerals
            if alive_time > 1 and current_mineral_distance < previous_mineral_distance:
                movement_towards_minerals += (
                    previous_mineral_distance - current_mineral_distance
                ) * 0.5

            previous_mineral_distance: float = current_mineral_distance

        # Enhanced fitness function
        # 1. Primary reward: Successfully mining minerals
        mineral_bonus = ship.minerals * 150

        # 2. Movement efficiency: Reward moving towards minerals
        movement_reward = movement_towards_minerals

        # 3. Survival time with diminishing returns
        survival_bonus = math.log(alive_time + 1) * 8

        # 4. Fuel efficiency: Don't waste fuel
        fuel_efficiency = (ship.fuel / 100.0) * 25

        # Combine fitness components
        genome.fitness = (
            mineral_bonus  # Main objective
            + movement_reward  # Encourage moving towards minerals
            + survival_bonus  # Basic survival
            + fuel_efficiency  # Don't waste fuel
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

        base_timeout = 3_000
        performance_bonus = min(
            2_000, ship.minerals * 500
        )  # Extra time for successful miners
        max_timeout = (
            base_timeout + performance_bonus
        )  # Ensure at least 10 thousand frames

        if asteroid_collision or out_of_fuel or alive_time >= max_timeout:
            # Penalty for collision or running out of fuel
            if asteroid_collision:
                genome.fitness -= 500
            elif out_of_fuel and ship.minerals == 0:
                genome.fitness -= 1000
            elif out_of_fuel:
                genome.fitness -= 250
            break

        if alive_time > 1_000 and ship.minerals == 0 and movement_towards_minerals < 10:
            genome.fitness -= 150  # Penalty for aimless wandering
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
    pygame.init()
    global screen, clock, WIDTH, HEIGHT
    WIDTH, HEIGHT = 800, 600
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
        neat.Checkpointer(1, filename_prefix=os.path.join(output_dir, "checkpoints/"))
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
