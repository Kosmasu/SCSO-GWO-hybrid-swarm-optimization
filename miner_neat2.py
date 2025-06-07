import pickle
import pygame
import os
import neat
import time
import copy

from custom_reporter import DataReporter
from data import WIDTH, HEIGHT, WHITE
from game_simulation import (
    SimulationConfig,
    run_neat_simulation,
)

from input_functions._4_full_radar import get_neat_inputs

if not pygame.get_init():
    pygame.init()


def run_simulation(genome, config, visualizer=None):
    """Run simulation using the modular game simulation"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create simulation config
    sim_config = SimulationConfig()
    sim_config.num_minerals = 5
    sim_config.num_asteroids = 10

    # Run the simulation
    result = run_neat_simulation(
        network=net,
        get_inputs_func=get_neat_inputs,
        config=sim_config,
        visualizer=visualizer,
        screen=screen if visualizer else None,
        clock=clock if visualizer else None,
    )

    # Set genome fitness
    genome.fitness = result.final_fitness

    return result.death_reason


FONT = pygame.font.SysFont(None, 36)


class TrainingVisualizer:
    def __init__(self):
        self.best_fitness = -float("inf")
        self.generation = 0
        self.start_time = time.time()
        # Add death statistics tracking
        self.death_stats = {
            "asteroid_collision": 0,
            "out_of_fuel": 0,
            "timeout": 0,
            "unknown": 0,
        }
        self.total_genomes_evaluated = 0

    def update_death_stats(self, death_reasons):
        """Update death statistics with results from a generation."""
        self.total_genomes_evaluated += len(death_reasons)
        for reason in death_reasons:
            if reason in self.death_stats:
                self.death_stats[reason] += 1
            else:
                self.death_stats["unknown"] += 1

    def get_death_percentages(self):
        """Get death reason percentages."""
        if self.total_genomes_evaluated == 0:
            return {reason: 0.0 for reason in self.death_stats}

        return {
            reason: (count / self.total_genomes_evaluated) * 100
            for reason, count in self.death_stats.items()
        }

    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")

        # Print death statistics
        death_percentages = self.get_death_percentages()
        print(f"Generation {self.generation} best: {best_genome.fitness:.1f}")
        print(
            f"💀 Death stats: Collision: {death_percentages['asteroid_collision']:.2f}% | "
            f"Fuel: {death_percentages['out_of_fuel']:.2f}% | "
            f"Timeout: {death_percentages['timeout']:.2f}%"
        )

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
    death_reasons = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        death_reason = run_simulation(
            genome, config, visualizer=None
        )  # No visualization during evaluation
        death_reasons.append(death_reason)

        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome

    # Update death statistics
    visualizer.update_death_stats(death_reasons)

    # Pass death reasons to config for reporter
    config.last_generation_death_reasons = death_reasons

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
