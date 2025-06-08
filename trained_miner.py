import pickle
import pygame
import neat
import os
from miner_neat2 import get_neat_inputs 
from game_simulation import SimulationConfig, run_neat_simulation

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_trained_model(filename):
    """Load a saved genome and config"""
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"Model loaded from {filename}")
    return save_data


class GameVisualizer:
    """Simple visualizer for trained model display"""
    def __init__(self):
        self.start_time = 0
    
    def draw_stats(self, screen, fitness, minerals, fuel):
        font = pygame.font.SysFont(None, 36)
        alive_time = int(pygame.time.get_ticks() / 1000)
        
        stats = [
            f"Minerals: {minerals}",
            f"Alive Time: {alive_time}",
            f"Fuel: {fuel:.1f}",
            f"Score: {fitness:.1f}",
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 25))


def run_with_trained_model():
    # Initialize pygame
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trained Miner Ship")
    clock = pygame.time.Clock()

    # Create config
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Load the trained model
    genome = load_trained_model('output/neat/20250608-001402/best_genome_gen_1061_fitness_68944.3.pkl')
    
    # Create the neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create simulation config
    sim_config = SimulationConfig()
    sim_config.num_minerals = 5
    sim_config.num_asteroids = 10
    sim_config.visualization_fps = 60
    
    # Create visualizer
    visualizer = GameVisualizer()
    
    # Run the simulation with visualization
    result = run_neat_simulation(
        network=net,
        get_inputs_func=get_neat_inputs,
        config=sim_config,
        visualizer=visualizer,
        screen=screen,
        clock=clock
    )
    
    print(f"Simulation ended: {result.death_reason}")
    print(f"Final score: {result.final_fitness:.1f}")
    print(f"Minerals collected: {result.minerals_collected}")
    print(f"Survived for: {result.alive_frames} frames")


if __name__ == "__main__":
    run_with_trained_model()