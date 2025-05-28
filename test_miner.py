import math
import pickle
import time
import neat
import pygame
from data import BLACK, HEIGHT, WIDTH
from game import Asteroid, Mineral, Spaceship
from miner_neat2 import TrainingVisualizer, get_neat_inputs


WINNER_DIR = "output/neat/winner.pkl"


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
        else:
            ship.velocity_x = 0
            ship.velocity_y = 0

        if output[2] > 0.5:
            ship.mine(minerals)
            if len(minerals) < 3:
                minerals.extend(Mineral() for _ in range(2))

        # Move asteroids
        for asteroid in asteroids:
            asteroid.move()

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

        if (
            asteroid_collision
            or out_of_fuel
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

    run_simulation(loaded_winner, config, visualizer=TrainingVisualizer())
