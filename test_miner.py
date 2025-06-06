import math
import pickle
import time
import neat
import pygame
from data import BLACK, HEIGHT, WIDTH
from game import Asteroid, Mineral, Spaceship
from miner_neat2 import TrainingVisualizer, get_neat_inputs
from game_simulation import SimulationConfig, run_neat_simulation


WINNER_DIR = "output/neat/20250603-152148/best_genome_gen_24_fitness_4697.7.pkl"


def run_simulation(genome, config, visualizer=None):
    """Run simulation using the modular approach"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create simulation config
    sim_config = SimulationConfig()
    sim_config.num_minerals = 5
    sim_config.num_asteroids = 8
    sim_config.visualization_fps = 60

    # Run the simulation
    result = run_neat_simulation(
        network=net,
        get_inputs_func=get_neat_inputs,
        config=sim_config,
        visualizer=visualizer,
        screen=screen if visualizer else None,
        clock=clock if visualizer else None,
    )

    # Set genome fitness (with legacy compatibility for test fitness calculation)
    survival_bonus = result.alive_frames / 10
    fuel_gain_bonus = result.total_fuel_gain * 5
    genome.fitness = survival_bonus + fuel_gain_bonus

    # Apply collision penalty
    if result.death_reason == "asteroid_collision":
        genome.fitness -= 50

    return result


if __name__ == "__main__":
    global screen, clock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT - Space Miner Testing")
    clock = pygame.time.Clock()

    # Load the winner genome from a file
    with open(WINNER_DIR, "rb") as f:
        loaded_winner = pickle.load(f)

    # Create config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.txt",
    )

    config.visualizer = TrainingVisualizer()

    run_simulation(loaded_winner, config, visualizer=TrainingVisualizer())
