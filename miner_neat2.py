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
    radar_scan,
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

    N_RADAR_DIRECTIONS = 16
    MAX_RADAR_RANGE = 300.0
    # Radar Scan Asteroids (16 inputs_value)
    radar_scan_results = radar_scan(
        ship, asteroids, n_directions=N_RADAR_DIRECTIONS, max_range=MAX_RADAR_RANGE
    )
    
    # Normalize radar distances (0 = max range/no obstacle, 1 = touching/collision)
    normalized_radar = [1.0 - (result.distance / MAX_RADAR_RANGE) for result in radar_scan_results]

    inputs_value.extend(normalized_radar)
    
    # Generate explanations for each radar direction (relative to ship heading)
    for i in range(N_RADAR_DIRECTIONS):
        angle_deg = i * (360 / N_RADAR_DIRECTIONS)
        inputs_explanation.append(f"Asteroid Radar {angle_deg:.0f}° relative (normalized inverse distance)")
    
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Radar Scan"
    )

    # Radar Scan Mineral (16 inputs_value)
    radar_scan_results = radar_scan(
        ship, minerals, n_directions=N_RADAR_DIRECTIONS, max_range=MAX_RADAR_RANGE
    )
    
    # Normalize radar distances (0 = max range/no obstacle, 1 = touching/collision)
    normalized_radar = [1.0 - (result.distance / MAX_RADAR_RANGE) for result in radar_scan_results]

    inputs_value.extend(normalized_radar)
    
    # Generate explanations for each radar direction (relative to ship heading)
    for i in range(N_RADAR_DIRECTIONS):
        angle_deg = i * (360 / N_RADAR_DIRECTIONS)
        inputs_explanation.append(f"Mineral Radar {angle_deg:.0f}° relative (normalized inverse distance)")
    
    assert len(inputs_value) == len(inputs_explanation), (
        "Inputs and explanations must match in length - Radar Scan"
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
        generation_bonus = min(
            20_000, config.visualizer.generation * 50
        )  # More time for later generations
        base_timeout_frame = 12_000 + generation_bonus

        # Mineral-based bonus (encourages mineral collection)
        mineral_bonus = min(18_000, ship.minerals * 1000)

        max_timeout_frame = base_timeout_frame + mineral_bonus
        # Cap at reasonable maximum
        max_timeout_frame = min(max_timeout_frame, 40_000)

        if (
            asteroid_collision
            or out_of_fuel
            or alive_frame_counter >= max_timeout_frame
        ):
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
