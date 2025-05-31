import math
import pickle
import time
import neat
import pygame
from data import BLACK, HEIGHT, WIDTH
from game import Asteroid, Mineral, Spaceship
from miner_neat2 import TrainingVisualizer, get_neat_inputs


WINNER_DIR = "output/neat/experiment-4/winner.pkl"


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
    total_fuel_gain = 0

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
        turn_output = output[0]
        if abs(turn_output - 0.5) > 0.05:  # Dead zone for turning
            turn_rate = (turn_output - 0.5) * 2 * 0.05
        else:
            turn_rate = 0
        ship.angle += turn_rate
        ship.angle = ship.angle % (2 * math.pi)

        # Bidirectional thrust (-1 = full backward, 0 = no thrust, 1 = full forward)
        # Thrust: more nuanced control
        thrust_output = output[1]

        # Only apply thrust if output is significantly different from 0.5 (neutral)
        if abs(thrust_output - 0.5) > 0.1:  # Dead zone for more stable behavior
            thrust_power = (thrust_output - 0.5) * 2  # Convert to -1 to 1
            # Scale thrust power more conservatively
            thrust_power *= 0.7  # Reduce maximum thrust
        else:
            thrust_power = 0  # No thrust in dead zone

        dx = thrust_power * ship.speed * math.cos(ship.angle)
        dy = thrust_power * ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)

        fuel_before = ship.fuel
        ship.mine(minerals)
        if len(minerals) < 3:
            minerals.extend(Mineral() for _ in range(2))
        fuel_gain = ship.fuel - fuel_before

        # Accumulate positive fuel gains only
        if fuel_gain > 0:
            total_fuel_gain += fuel_gain

        # Move asteroids
        for asteroid in asteroids:
            asteroid.move()

        # Enhanced fitness function
        # 1. Survival time with linear growth
        survival_bonus = alive_frame_counter / 4

        # 2. Mineral collection bonus
        mineral_bonus = ship.minerals * 100

        # Combine fitness components
        genome.fitness = (
            survival_bonus  # Main objective
            + mineral_bonus  # Encourage mineral collection
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
            clock.tick(60)

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


if __name__ == "__main__":
    global screen, clock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    # Load the winner genome from a file
    with open(WINNER_DIR, "rb") as f:
        loaded_winner = pickle.load(f)

    # Create and store visualizer in config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.txt",
    )

    config.visualizer = TrainingVisualizer()

    run_simulation(loaded_winner, config, visualizer=TrainingVisualizer())
