import pickle
import pygame
import math
import os
import neat
import time
import copy

from custom_reporter import DataReporter
from data import ASTEROID_MAX_SPEED, BLACK, WIDTH, HEIGHT, WHITE
from game import (
    Asteroid,
    Mineral,
    Spaceship,
    get_closest_asteroid_info,
    get_closest_mineral_info,
    radar_scan,
)

if not pygame.get_init():
    pygame.init()


def handle_steering(output: list[float]):
    """
    Handle steering based on the output from the neural network.
    The output is expected to be in the range [-1, 1] for turning left and right.
    """
    turn_output = output[0]
    if turn_output < -0.3:
        return ((turn_output + 0.3) / 0.7) * 0.15  # Full left
    elif turn_output >= -0.3 and turn_output <= 0.3:
        return 0  # No turn
    else:
        return ((turn_output - 0.3) / 0.7) * 0.15  # Full right


def handle_thrusting(output: list[float]):
    """
    Handle thrusting based on the output from the neural network.
    The output is expected to be in the range [-1, 1] for backward and forward thrust.
    """
    thrust_output = output[1]
    if thrust_output < -0.3:
        return ((thrust_output + 0.3) / 0.7) * 0.8  # Full backward
    elif thrust_output >= -0.3 and thrust_output <= 0.3:
        return 0  # No thrust
    else:
        return (thrust_output - 0.3) / 0.7  # Full forward


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

    # Top 1 Closest Asteroid (3 inputs_value)
    MAX_ASTEROID_SPEED_NORMAL = math.sqrt(ASTEROID_MAX_SPEED**2 + ASTEROID_MAX_SPEED**2)
    closest_asteroids = get_closest_asteroid_info(ship, asteroids, top_n=1)
    if closest_asteroids:
        closest_asteroid = closest_asteroids[0]
        inputs_value.extend(
            [
                max(
                    0.0, 1.0 - (closest_asteroid.distance / MAX_RADAR_RANGE)
                ),  # Normalize distance
                math.sin(closest_asteroid.relative_angle),  # Y component of angle
                math.cos(closest_asteroid.relative_angle),  # X component of angle
                math.sin(
                    closest_asteroid.velocity_angle
                ),  # Y component of velocity direction
                math.cos(
                    closest_asteroid.velocity_angle
                ),  # X component of velocity direction
                closest_asteroid.velocity_magnitude
                / MAX_ASTEROID_SPEED_NORMAL,  # Normalized velocity magnitude (assuming max ~5)
            ]
        )
    else:
        # Pad with safe default values if no asteroids are present
        inputs_value.extend([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    inputs_explanation.extend(
        [
            "Closest Asteroid Distance (normalized)",
            "Closest Asteroid Relative Angle Sin (normalized)",
            "Closest Asteroid Relative Angle Cos (normalized)",
            "Asteroid Relative Velocity Direction Sin",
            "Asteroid Relative Velocity Direction Cos",
            "Asteroid Relative Velocity Magnitude",
        ]
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


def run_simulation(genome, config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship()

    minerals: list[Mineral] = []
    asteroids: list[Asteroid] = []

    # Generate initial asteroids
    generation = config.visualizer.generation if visualizer else 0
    if generation < 10: 
        # Pure movement and mining learning - NO asteroids
        MINERAL_COUNT = 3
        ASTEROID_COUNT = 0  # No asteroids at all
        MAX_FRAME = 2000
        ASTEROID_SPEED_MULTIPLIER = 0
        DIFFICULTY_MULTIPLIER = 1.0  # Full fitness signal for learning basics
    elif generation < 30:
        # Learn basic movement with stationary obstacles
        MINERAL_COUNT = 3
        ASTEROID_COUNT = 1
        MAX_FRAME = 3000
        ASTEROID_SPEED_MULTIPLIER = 0  # Stationary asteroids
        DIFFICULTY_MULTIPLIER = 1.0
    elif generation < 60:
        # Introduce very slow moving asteroids
        MINERAL_COUNT = 3
        ASTEROID_COUNT = 2
        MAX_FRAME = 4000
        ASTEROID_SPEED_MULTIPLIER = 0.1  # Very slow
        DIFFICULTY_MULTIPLIER = 1.0
    elif generation < 100:
        # Slightly faster asteroids
        MINERAL_COUNT = 3
        ASTEROID_COUNT = 3
        MAX_FRAME = 5000
        ASTEROID_SPEED_MULTIPLIER = 0.25
        DIFFICULTY_MULTIPLIER = 1.0
    elif generation < 150:
        # More asteroids, still manageable speed
        MINERAL_COUNT = 3
        ASTEROID_COUNT = 4
        MAX_FRAME = 6000
        ASTEROID_SPEED_MULTIPLIER = 0.35
        DIFFICULTY_MULTIPLIER = 1.0
    elif generation < 200:
        # Increase asteroid count
        MINERAL_COUNT = 3
        ASTEROID_COUNT = 5
        MAX_FRAME = 7000
        ASTEROID_SPEED_MULTIPLIER = 0.45
        DIFFICULTY_MULTIPLIER = 1.0
    elif generation < 300:
        # More challenging
        MINERAL_COUNT = 4
        ASTEROID_COUNT = 6
        MAX_FRAME = 8000
        ASTEROID_SPEED_MULTIPLIER = 0.6
        DIFFICULTY_MULTIPLIER = 1.0
    elif generation < 400:
        # Near full difficulty
        MINERAL_COUNT = 4
        ASTEROID_COUNT = 7
        MAX_FRAME = 10000
        ASTEROID_SPEED_MULTIPLIER = 0.8
        DIFFICULTY_MULTIPLIER = 1.0
    else:
        # Full game mechanics
        MINERAL_COUNT = 5
        ASTEROID_COUNT = 8
        MAX_FRAME = 100_000
        ASTEROID_SPEED_MULTIPLIER = 1.0
        DIFFICULTY_MULTIPLIER = 1.0

    # Initialize minerals
    for _ in range(MINERAL_COUNT):
        minerals.append(Mineral())

    # Initialize asteroids
    for _ in range(ASTEROID_COUNT):
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
    total_fuel_gain = 0
    backward_movement_counter = 0
    total_movement_counter = 0

    # Add spinning tracking variables
    total_angle_change = 0
    significant_turn_counter = 0
    total_turn_counter = 0
    previous_angle = ship.angle

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

        ship.angle += handle_steering(output)
        ship.angle = ship.angle % (2 * math.pi)
        thrust_power = handle_thrusting(output)

        # Track backward movement
        if thrust_power != 0:
            total_movement_counter += 1
            if thrust_power < 0:  # Backward movement
                backward_movement_counter += 1

        dx = thrust_power * ship.speed * math.cos(ship.angle)
        dy = thrust_power * ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)

        fuel_before = ship.fuel
        ship.mine(minerals)
        if len(minerals) < MINERAL_COUNT:
            minerals.append(Mineral())
        fuel_gain = ship.fuel - fuel_before

        # Accumulate positive fuel gains only
        if fuel_gain > 0:
            total_fuel_gain += fuel_gain

        # Move asteroids
        for asteroid in asteroids:
            asteroid.move(ASTEROID_SPEED_MULTIPLIER)

        # Enhanced fitness function with generation-aware scaling
        # 1. Survival time with linear growth
        survival_bonus = alive_frame_counter / 4

        # 2. Fuel gain bonus (more important in early generations)
        fuel_gain_bonus = total_fuel_gain * (10 if generation < 50 else 5)

        # 3. Mineral collection bonus (scaled by generation)
        mineral_collection_bonus = ship.minerals * (200 if generation < 100 else 100)

        # 4. Movement encouragement (early generations only)
        movement_bonus = 0
        if generation < 50 and total_movement_counter > 0:
            # Encourage any movement in early generations
            movement_bonus = min(100, total_movement_counter * 0.5)

        # 5. Asteroid avoidance bonus (only when asteroids are present)
        avoidance_bonus = 0
        if ASTEROID_COUNT > 0 and alive_frame_counter > 100:
            # Bonus for surviving longer with asteroids present
            avoidance_bonus = (alive_frame_counter - 100) * 0.2

        # 6. Backward movement penalty (reduced in early generations)
        backward_penalty = 0
        if total_movement_counter > 0 and generation > 30:
            backward_ratio = backward_movement_counter / total_movement_counter
            if backward_ratio > 0.4:  # More lenient threshold
                excess_backward = backward_ratio - 0.4
                backward_penalty = excess_backward * (100 if generation < 100 else 300)

        # 7. Spinning penalty (only after movement is learned)
        spinning_penalty = 0
        if alive_frame_counter > 100 and generation > 50:
            avg_angle_change_per_frame = total_angle_change / alive_frame_counter
            if avg_angle_change_per_frame > 0.03:  # More lenient threshold
                excess_spinning = avg_angle_change_per_frame - 0.03
                spinning_penalty = excess_spinning * 2000

        # Combine fitness components
        genome.fitness = (
            survival_bonus
            + fuel_gain_bonus
            + mineral_collection_bonus
            + movement_bonus  # New: encourage movement in early stages
            + avoidance_bonus  # New: reward asteroid avoidance
            - backward_penalty
            - spinning_penalty
        ) * DIFFICULTY_MULTIPLIER

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
            clock.tick(60)

        if asteroids:
            closest_asteroid = min(
                (a for a in asteroids),
                key=lambda a: math.hypot(ship.x - a.x, ship.y - a.y),
            )
            # Termination conditions
            asteroid_collision = (
                math.hypot(ship.x - closest_asteroid.x, ship.y - closest_asteroid.y)
                < ship.radius + closest_asteroid.radius
            )
        else:
            asteroid_collision = False
        out_of_fuel = ship.fuel <= 0

        base_timeout_frame = 12_000
        # Adaptive timeout based on performance and generation
        generation_bonus = config.visualizer.generation * 200

        # Mineral-based bonus (encourages mineral collection)
        mineral_bonus = min(18_000, ship.minerals * 1000)

        max_timeout_frame = min(
            base_timeout_frame + mineral_bonus + generation_bonus, MAX_FRAME
        )

        if (
            asteroid_collision
            or out_of_fuel
            or alive_frame_counter >= max_timeout_frame
        ):
            if asteroid_collision:
                genome.fitness = max(0, genome.fitness - 200)
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

    # Manual best genome tracking to avoid NEAT bug
    if not hasattr(config, "manual_best_genome"):
        config.manual_best_genome = None
        config.manual_best_fitness = -float("inf")

    if best_fitness > config.manual_best_fitness:
        config.manual_best_genome = copy.deepcopy(
            best_in_generation
        )  # Deep copy to avoid reference issues
        config.manual_best_fitness = best_fitness
        print(f"🏆 NEW OVERALL BEST: {best_fitness:.1f}")

        # Save the new best genome immediately
        best_genome_filename = (
            f"best_genome_gen_{visualizer.generation}_fitness_{best_fitness:.1f}.pkl"
        )
        best_genome_path = os.path.join(config.output_dir, best_genome_filename)
        with open(best_genome_path, "wb") as f:
            pickle.dump(config.manual_best_genome, f)
        print(f"💾 Best genome saved: {best_genome_filename}")

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

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Set up visualizer
    visualizer = TrainingVisualizer()
    config.visualizer = visualizer
    config.output_dir = (
        output_dir  # Store output_dir in config for access in eval_genomes
    )

    # Create or restore population
    if continue_from_checkpoint:
        # Find the latest checkpoint
        checkpoint_files = [
            f for f in os.listdir(checkpoints_dir) if f.startswith("neat-checkpoint-")
        ]
        if checkpoint_files:
            # Sort by checkpoint number and get the latest
            checkpoint_files.sort(key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = os.path.join(checkpoints_dir, checkpoint_files[-1])
            print(f"🔄 Restoring from checkpoint: {latest_checkpoint}")
            population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)

            # Update config with restored population's generation
            visualizer.generation = population.generation
        else:
            print("⚠️  No checkpoint files found, starting fresh...")
            population = neat.Population(config)
            continue_from_checkpoint = False  # No actual checkpoint to continue from
    else:
        # Create a new population
        population = neat.Population(config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            25, filename_prefix=os.path.join(checkpoints_dir, "neat-checkpoint-")
        )
    )

    # Add our clean data reporter with checkpoint flag
    data_reporter = DataReporter(
        output_dir=output_dir,
        config_file_path=config_file,
        continue_from_checkpoint=continue_from_checkpoint,
    )
    population.add_reporter(data_reporter)

    # Run NEAT
    try:
        population.run(eval_genomes, 100_000)

        # Save final summary
        data_reporter.save_final_summary()

        # Fix: Use manual tracking or population.best_genome
        winner = getattr(config, "manual_best_genome", None) or population.best_genome

        if not winner:
            raise ValueError("No winner genome found after training.")

        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")

        # Test the winner
        print("\nTesting winner...")
        run_simulation(winner, config, visualizer=config.visualizer)

        # Save the winner
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
