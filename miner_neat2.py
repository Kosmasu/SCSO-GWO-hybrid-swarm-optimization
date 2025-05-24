import pygame
import random
import math
import os
import neat
import time
from pydantic import BaseModel, Field
# POOR FITNESS FUNCTION - spin on own axis

# Initialize pygame
# Note: pygame, random, and math are assumed to be imported at the top of the file,
# as they are used by the classes below. e.g.:
# import pygame
# import random
# import math

# Initialize pygame and define globals with type hints
pygame.init()
WIDTH: int = 800
HEIGHT: int = 600
screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT - Space Miner Training")
clock: pygame.time.Clock = pygame.time.Clock()

# Colors with type hints
BLACK: tuple[int, int, int] = (0, 0, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (255, 0, 0)
GREEN: tuple[int, int, int] = (0, 255, 0)
BLUE: tuple[int, int, int] = (0, 0, 255)
YELLOW: tuple[int, int, int] = (255, 255, 0)

# Game Classes (modified to use Pydantic BaseModel)
# Original class order is maintained.

class Spaceship(BaseModel):
    # Attributes with type hints and Pydantic Field for default values/factories
    x: float = Field(default_factory=lambda: float(WIDTH // 2))
    y: float = Field(default_factory=lambda: float(HEIGHT // 2))
    speed: float = 5.0
    angle: float = 0.0
    fuel: float = 100.0
    minerals: int = 0
    radius: int = 15

    # Methods with type hints for parameters and return types
    def move(self, dx: float, dy: float) -> None:
        if self.fuel > 0:
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.fuel -= 0.1  # fuel is float

    def mine(self, minerals: list['Mineral']) -> None:  # Using string literal for forward reference to Mineral
        for mineral_obj in minerals[:]:  # Iterate over a copy
            dist: float = math.hypot(self.x - mineral_obj.x, self.y - mineral_obj.y)
            # Ensure mineral_obj attributes are accessed (Pydantic model fields)
            if dist < self.radius + mineral_obj.radius:
                minerals.remove(mineral_obj)
                self.minerals += 1
                self.fuel = min(100.0, self.fuel + 10.0) # Ensure float arithmetic for fuel

    def draw(self) -> None:
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)
        
        # Calculate points for the ship's triangular nose
        nose_x: float = self.x + self.radius * math.cos(self.angle)
        nose_y: float = self.y + self.radius * math.sin(self.angle)
        left_fin_x: float = self.x + self.radius * math.cos(self.angle + 2.5) 
        left_fin_y: float = self.y + self.radius * math.sin(self.angle + 2.5)
        right_fin_x: float = self.x + self.radius * math.cos(self.angle - 2.5)
        right_fin_y: float = self.y + self.radius * math.sin(self.angle - 2.5)
        
        points: list[tuple[float, float]] = [
            (nose_x, nose_y),
            (left_fin_x, left_fin_y),
            (right_fin_x, right_fin_y)
        ]
        pygame.draw.polygon(screen, WHITE, points)

    class Config:
        arbitrary_types_allowed = True # Useful for Pydantic models with complex field types

class Mineral(BaseModel):
    x: int = Field(default_factory=lambda: random.randint(20, WIDTH - 20))
    y: int = Field(default_factory=lambda: random.randint(20, HEIGHT - 20))
    radius: int = 10

    def draw(self) -> None:
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.radius)

    class Config:
        arbitrary_types_allowed = True

class Asteroid(BaseModel):
    x: float = Field(default_factory=lambda: float(random.randint(0, WIDTH)))
    y: float = Field(default_factory=lambda: float(random.randint(0, HEIGHT)))
    radius: int = Field(default_factory=lambda: random.randint(15, 30))
    speed_x: float = Field(default_factory=lambda: random.uniform(-2, 2))
    speed_y: float = Field(default_factory=lambda: random.uniform(-2, 2))

    def move(self) -> None:
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self) -> None:
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

    class Config:
        arbitrary_types_allowed = True

# Note: If you are not using `from __future__ import annotations` at the top of your file,
# or if you are using an older version of Pydantic (V1), you might need to manually
# update forward references for models that use them, e.g., by calling:
# Spaceship.update_forward_refs()
# This is typically done after all relevant model definitions.
# With Pydantic V2 and modern Python, this is often handled automatically.

def run_simulation(genome, config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    alive_time = 0
    
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Find closest objects
        closest_mineral = min((m for m in minerals), 
                            key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), 
                            default=None)
        closest_asteroid = min((a for a in asteroids), 
                              key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y))
        
        # Get inputs (handle case where all minerals are collected)
        inputs = [
            math.hypot(ship.x - closest_mineral.x)/WIDTH if closest_mineral else 0,
            math.atan2(closest_mineral.y-ship.y, closest_mineral.x-ship.x)/math.pi if closest_mineral else 0,
            math.hypot(ship.x - closest_asteroid.x)/WIDTH,
            ship.fuel / 100.0
        ]
        
        # Get actions from network
        output = net.activate(inputs)
        
        # Execute actions
        ship.angle += (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
        if output[1] > 0.5:  # Thrust
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            ship.move(dx, dy)
        if output[2] > 0.5:  # Mine
            ship.mine(minerals)
            if len(minerals) < 3:  # Replenish minerals
                minerals.extend(Mineral() for _ in range(2))
        
        # Calculate fitness - reward both survival and mining
        genome.fitness = ship.minerals * 10 + alive_time * 0.01  # Reduced time bonus
        
        # Visualization
        if visualizer:
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw()
            for asteroid in asteroids:
                asteroid.move()
                asteroid.draw()
            ship.draw()
            visualizer.draw_stats(screen, genome.fitness, ship.minerals, ship.fuel)
            pygame.display.flip()
            clock.tick(30)
        
        # Termination conditions
        asteroid_collision = math.hypot(ship.x-closest_asteroid.x, ship.y-closest_asteroid.y) < ship.radius + closest_asteroid.radius
        out_of_fuel = ship.fuel <= 0
        no_minerals_left = not minerals and ship.minerals == 0
        
        if asteroid_collision or out_of_fuel or no_minerals_left or alive_time >= 5000:
            break

class TrainingVisualizer:
    def __init__(self):
        self.best_fitness = -float('inf')
        self.generation = 0
        self.start_time = time.time()
        self.font = pygame.font.SysFont(None, 36)
        
    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")
        print(f"Generation {self.generation} best: {best_genome.fitness:.1f}")

    def draw_stats(self, screen, fitness, minerals, fuel):
        stats = [
            f"Gen: {self.generation}",
            f"Fitness: {fitness:.1f}",
            f"Best: {self.best_fitness:.1f}",
            f"Minerals: {minerals}",
            f"Fuel: {fuel:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 40))

def eval_genomes(genomes, config):
    visualizer = config.visualizer
    
    # First evaluate all genomes to find the best
    best_in_generation = None
    best_fitness = -float('inf')
    
    for genome_id, genome in genomes:
        run_simulation(genome, config, visualizer=None)  # No visualization during evaluation
        print(genome_id,genome.fitness)
        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome
    
    # Update visualizer with this generation's results
    visualizer.update_generation(best_in_generation)
    
    # Visualize the best genome from this generation
    if best_in_generation:
        print(f"Displaying generation {visualizer.generation} best (Fitness: {best_fitness:.1f})")
        run_simulation(best_in_generation, config, visualizer=visualizer)  # With visualization


def run_neat(config_file):
    # Initialize pygame
    pygame.init()
    global screen, clock, WIDTH, HEIGHT
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    
    # Create and store visualizer in config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer()
    
    # Create population
    population = neat.Population(config)
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run NEAT
    try:
        winner = population.run(eval_genomes, 50)
        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)