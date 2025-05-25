import pickle
import pygame
import math
import os
import neat
import time

from data import WIDTH, HEIGHT, WHITE
from game import Asteroid, Mineral, Spaceship
# Initialize pygame and define globals with type hints
pygame.init()
screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT - Space Miner Training")
clock: pygame.time.Clock = pygame.time.Clock()



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
    fuel_consumed = 0
    initial_fuel = ship.fuel
    last_x, last_y = ship.x, ship.y  # Track movement for exploration bonus
    
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Find closest objects - we'll replace this single mineral lookup with multiple
        closest_mineral = min((m for m in minerals), 
                            key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), 
                            default=None)
        
        # Calculate wrapped distances to account for screen teleportation
        def wrapped_distance(obj_x, obj_y, ship_x, ship_y):
            # Calculate direct distance
            direct_dx = obj_x - ship_x
            direct_dy = obj_y - ship_y
            
            # Calculate wrapped distances in x and y
            wrapped_dx = min(abs(direct_dx), WIDTH - abs(direct_dx)) * (1 if direct_dx >= 0 else -1)
            wrapped_dy = min(abs(direct_dy), HEIGHT - abs(direct_dy)) * (1 if direct_dy >= 0 else -1)
            
            # Return actual minimum distance considering wrapping
            return math.hypot(wrapped_dx, wrapped_dy)
        
        # Get wrapped angle (considering teleportation)
        def wrapped_angle(obj_x, obj_y, ship_x, ship_y):
            # Find the shortest path considering wrapping
            dx = obj_x - ship_x
            dy = obj_y - ship_y
            
            # Adjust for wrap-around in x
            if abs(dx) > WIDTH / 2:
                dx = -1 * (WIDTH - abs(dx)) * (1 if dx >= 0 else -1)
                
            # Adjust for wrap-around in y
            if abs(dy) > HEIGHT / 2:
                dy = -1 * (HEIGHT - abs(dy)) * (1 if dy >= 0 else -1)
                
            return math.atan2(dy, dx)
            
        # Calculate asteroid threat levels with wrapped distances
        asteroid_threats = []
        for asteroid in asteroids:
            # Calculate actual distance considering screen wrapping
            wrapped_dist = wrapped_distance(asteroid.x, asteroid.y, ship.x, ship.y)
            wrapped_ang = wrapped_angle(asteroid.x, asteroid.y, ship.x, ship.y)
            
            # Calculate potential future positions after wrapping
            # If asteroid is near edge, consider it might wrap
            is_near_edge_x = asteroid.x < 50 or asteroid.x > WIDTH - 50
            is_near_edge_y = asteroid.y < 50 or asteroid.y > HEIGHT - 50
            is_near_edge = is_near_edge_x or is_near_edge_y
            
            # Calculate predicted position after 15 frames
            future_x = (asteroid.x + asteroid.speed_x * 15) % WIDTH
            future_y = (asteroid.y + asteroid.speed_y * 15) % HEIGHT
            
            # Will the asteroid wrap around in the next few frames?
            will_wrap_x = (asteroid.x < 50 and asteroid.speed_x < 0) or (asteroid.x > WIDTH - 50 and asteroid.speed_x > 0)
            will_wrap_y = (asteroid.y < 50 and asteroid.speed_y < 0) or (asteroid.y > HEIGHT - 50 and asteroid.speed_y > 0)
            will_wrap = will_wrap_x or will_wrap_y
            
            # Calculate future wrapped distance
            future_wrapped_dist = wrapped_distance(future_x, future_y, ship.x, ship.y)
            
            # Calculate asteroid's speed and direction
            speed_magnitude = math.hypot(asteroid.speed_x, asteroid.speed_y)
            movement_angle = math.atan2(asteroid.speed_y, asteroid.speed_x)
            
            # Store all the information
            asteroid_threats.append({
                'asteroid': asteroid,
                'dist': wrapped_dist,
                'angle': wrapped_ang,
                'future_dist': future_wrapped_dist,
                'speed': speed_magnitude,
                'movement_angle': movement_angle,
                'near_edge': is_near_edge,
                'will_wrap': will_wrap,
                'radius': asteroid.radius
            })
        
        # Sort by true distance (wrapped distance)
        asteroid_threats.sort(key=lambda x: x['dist'])
        
        # Process minerals with wrapped distances (similar to asteroids)
        mineral_targets = []
        for mineral in minerals:
            # Calculate actual distance considering screen wrapping
            wrapped_dist = wrapped_distance(mineral.x, mineral.y, ship.x, ship.y)
            wrapped_ang = wrapped_angle(mineral.x, mineral.y, ship.x, ship.y)
            
            # Calculate if mineral is near an edge (for edge-crossing awareness)
            is_near_edge_x = mineral.x < 50 or mineral.x > WIDTH - 50
            is_near_edge_y = mineral.y < 50 or mineral.y > HEIGHT - 50
            is_near_edge = is_near_edge_x or is_near_edge_y
            
            # Store mineral information
            mineral_targets.append({
                'mineral': mineral,
                'dist': wrapped_dist,
                'angle': wrapped_ang,
                'near_edge': is_near_edge
            })
        
        # Sort minerals by distance
        mineral_targets.sort(key=lambda x: x['dist'])
        
        # Enhanced inputs for better spatial awareness
        inputs = []
        
        # Process top 3 closest minerals for better resource awareness
        for i in range(min(3, len(mineral_targets))):
            target = mineral_targets[i]
            mineral = target['mineral']
            
            # Calculate normalized distance
            mineral_dist = target['dist'] / math.sqrt(WIDTH**2 + HEIGHT**2)
            
            # Calculate relative angle to the mineral
            mineral_angle = target['angle']
            mineral_rel_angle = (mineral_angle - ship.angle + math.pi) % (2 * math.pi) - math.pi
            
            # Edge awareness for minerals
            near_edge = 1.0 if target['near_edge'] else 0.0
            
            # Add mineral information to inputs
            inputs.extend([
                mineral_dist,                   # Distance to mineral
                math.cos(mineral_rel_angle),    # Direction to mineral (x component)
                math.sin(mineral_rel_angle),    # Direction to mineral (y component)
                near_edge                       # Is mineral near edge?
            ])
        
        # If fewer than 3 minerals, pad with default values
        for i in range(3 - min(3, len(mineral_targets))):
            inputs.extend([1.0, 0.0, 0.0, 0.0])  # No mineral: max distance, no direction, not near edge
        
        # Process top 3 closest asteroids for better threat awareness
        for i in range(min(3, len(asteroid_threats))):
            threat = asteroid_threats[i]
            asteroid = threat['asteroid']
            
            # Basic position information
            norm_dist = threat['dist'] / math.sqrt(WIDTH**2 + HEIGHT**2)
            asteroid_rel_angle = (threat['angle'] - ship.angle + math.pi) % (2 * math.pi) - math.pi
            
            # Asteroid size (normalized)
            norm_radius = threat['radius'] / 30.0  # Max radius is around 30
            
            # Asteroid speed and direction
            speed_magnitude = threat['speed'] / 3.0  # Normalize speed
            movement_rel_angle = (threat['movement_angle'] - ship.angle + math.pi) % (2 * math.pi) - math.pi
            
            # Future position
            future_dist = threat['future_dist'] / math.sqrt(WIDTH**2 + HEIGHT**2)
            
            # Edge and wrap detection
            near_edge = 1.0 if threat['near_edge'] else 0.0
            will_wrap = 1.0 if threat['will_wrap'] else 0.0
            
            # Add all asteroid information to inputs
            inputs.extend([
                norm_dist,                    # Current distance
                math.cos(asteroid_rel_angle), # Direction to asteroid
                math.sin(asteroid_rel_angle), 
                norm_radius,                  # Size of asteroid
                speed_magnitude,              # How fast it's moving
                math.cos(movement_rel_angle), # Direction it's moving
                math.sin(movement_rel_angle),
                future_dist,                  # Predicted future distance
                near_edge,                    # Is asteroid near the edge?
                will_wrap                     # Will asteroid wrap soon?
            ])
        
        # If fewer than 3 asteroids, pad with safe values
        for i in range(3 - min(3, len(asteroid_threats))):
            inputs.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        
        # Ship state - add edge proximity awareness
        ship_near_edge_x = ship.x < 50 or ship.x > WIDTH - 50
        ship_near_edge_y = ship.y < 50 or ship.y > HEIGHT - 50
        
        inputs.extend([
            ship.fuel / 100.0,
            ship.x / WIDTH,
            ship.y / HEIGHT,
            math.cos(ship.angle),
            math.sin(ship.angle),
            1.0 if ship_near_edge_x else 0.0,  # Is ship near horizontal edge?
            1.0 if ship_near_edge_y else 0.0   # Is ship near vertical edge?
        ])
        
        # Enhanced boundary distances with wrap awareness
        left_wall = ship.x
        right_wall = WIDTH - ship.x
        top_wall = ship.y
        bottom_wall = HEIGHT - ship.y
        
        # If we're closer to the opposite edge via wrapping, use that distance
        if left_wall > right_wall:
            left_wall, right_wall = right_wall, left_wall
        if top_wall > bottom_wall:
            top_wall, bottom_wall = bottom_wall, top_wall
            
        inputs.extend([
            left_wall / WIDTH,   # Distance to closest horizontal wall
            right_wall / WIDTH,  # Distance to farthest horizontal wall
            top_wall / HEIGHT,   # Distance to closest vertical wall
            bottom_wall / HEIGHT # Distance to farthest vertical wall
        ])
        
        # Additional mineral awareness (count nearby minerals)
        nearby_minerals = sum(1 for m in minerals if wrapped_distance(m.x, m.y, ship.x, ship.y) < 100)
        inputs.append(nearby_minerals / 5.0)
        
        # Wrap hazard detection - count asteroids that might appear from edges
        wrap_hazards = sum(1 for t in asteroid_threats if t['will_wrap'] and t['future_dist'] < 100)
        inputs.append(wrap_hazards / len(asteroids))
        
        # Movement tracking
        movement_dist = math.hypot(ship.x - last_x, ship.y - last_y)
        last_x, last_y = ship.x, ship.y
        
        # Get actions from network
        output = net.activate(inputs)
        
        # Track fuel consumption
        old_fuel = ship.fuel
        
        # Execute actions with improved mapping
        ship.angle += (output[0] * 2 - 1) * 0.15  # Turn (-1 to 1) - increased sensitivity
        
        # Evasive action if asteroid is too close or about to wrap and appear close
        closest_threat = asteroid_threats[0] if asteroid_threats else None
        is_in_danger = False
        
        if closest_threat:
            is_in_danger = (closest_threat['dist'] < 60) or (closest_threat['will_wrap'] and closest_threat['future_dist'] < 80)
        
        if is_in_danger and closest_threat:
            # Calculate escape vector (away from asteroid or its future position if wrapping)
            escape_x, escape_y = ship.x, ship.y
            threat_x, threat_y = closest_threat['asteroid'].x, closest_threat['asteroid'].y
            
            # If asteroid will wrap, predict where it will appear
            if closest_threat['will_wrap']:
                future_x = (threat_x + closest_threat['asteroid'].speed_x * 5) % WIDTH
                future_y = (threat_y + closest_threat['asteroid'].speed_y * 5) % HEIGHT
                
                # Determine if the future position is closer after wrapping
                if wrapped_distance(future_x, future_y, ship.x, ship.y) < closest_threat['dist']:
                    threat_x, threat_y = future_x, future_y
            
            # Calculate escape angle away from the threat
            escape_angle = wrapped_angle(escape_x, escape_y, threat_x, threat_y)
            
            # Turn towards escape direction
            angle_diff = (escape_angle - ship.angle + math.pi) % (2 * math.pi) - math.pi
            ship.angle += 0.2 * (1 if angle_diff > 0 else -1)  # Faster turning in emergency
            
            # Emergency thrust
            dx = ship.speed * 1.5 * math.cos(ship.angle)  # Boost speed for escape
            dy = ship.speed * 1.5 * math.sin(ship.angle)
            ship.move(dx, dy)
        else:
            # Normal thrust behavior
            if output[1] > 0.3:  # Thrust (lowered threshold)
                dx = ship.speed * math.cos(ship.angle)
                dy = ship.speed * math.sin(ship.angle)
                ship.move(dx, dy)
        
        # Only mine when relatively safe
        if output[2] > 0.5 and (not is_in_danger):  # Safe mining
            ship.mine(minerals)
            if len(minerals) < 3:  # Replenish minerals
                minerals.extend(Mineral() for _ in range(2))
        
        fuel_consumed += old_fuel - ship.fuel
        
        # Enhanced fitness function
        mineral_bonus = ship.minerals * 75  # Increased mineral reward
        survival_bonus = alive_time * 0.15
        fuel_efficiency = max(0, (initial_fuel - fuel_consumed) * 0.3)  # Reward fuel conservation
        
        # Exploration bonus (reward movement)
        exploration_bonus = min(15, movement_dist * 0.8) if alive_time > 50 else 0
        
        # Safety bonus (reward staying away from asteroids)
        safety_bonus = 0
        if closest_threat:
            safety_bonus = min(20, closest_threat['dist'] * 0.2)
        
        # Edge avoidance bonus (slightly discourage edge proximity unless necessary)
        edge_avoidance = 0
        if not any(t['near_edge'] for t in asteroid_threats[:2]) and (ship_near_edge_x or ship_near_edge_y):
            edge_avoidance = -5  # Small penalty for unnecessarily being near edge
        
        genome.fitness = mineral_bonus + survival_bonus + fuel_efficiency + exploration_bonus + safety_bonus + edge_avoidance
        
        # Visualization
        if visualizer:
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw()
                
            for asteroid in asteroids:
                asteroid.move()
                asteroid.draw()
                
                # Visualize asteroid trajectory and wrapping
                if visualizer:
                    # Show trajectory line
                    future_x = (asteroid.x + asteroid.speed_x * 15) % WIDTH
                    future_y = (asteroid.y + asteroid.speed_y * 15) % HEIGHT
                    
                    # Draw normal trajectory
                    pygame.draw.line(screen, (100, 100, 100), 
                                    (int(asteroid.x), int(asteroid.y)), 
                                    (int(asteroid.x + asteroid.speed_x * 15), int(asteroid.y + asteroid.speed_y * 15)), 1)
                    
                    # If it will wrap, show where it will appear
                    will_wrap_x = (asteroid.x < 50 and asteroid.speed_x < 0) or (asteroid.x > WIDTH - 50 and asteroid.speed_x > 0)
                    will_wrap_y = (asteroid.y < 50 and asteroid.speed_y < 0) or (asteroid.y > HEIGHT - 50 and asteroid.speed_y > 0)
                    
                    if will_wrap_x or will_wrap_y:
                        pygame.draw.line(screen, (200, 100, 100), 
                                        (int(asteroid.x), int(asteroid.y)),
                                        (int(future_x), int(future_y)), 2)
                        # Draw a circle where it will appear
                        pygame.draw.circle(screen, (255, 100, 100), (int(future_x), int(future_y)), 5, 1)
                    
            ship.draw()
            
            # Visualize danger zone
            if is_in_danger:
                pygame.draw.circle(screen, (255, 50, 50), (int(ship.x), int(ship.y)), 60, 2)
                
            # Visualize edge zone for ship
            if ship_near_edge_x or ship_near_edge_y:
                pygame.draw.circle(screen, (50, 50, 255), (int(ship.x), int(ship.y)), 20, 1)
                
            visualizer.draw_stats(screen, genome.fitness, ship.minerals, ship.fuel)
            pygame.display.flip()
            clock.tick(30)
        
        # Update asteroids
        for asteroid in asteroids:
            asteroid.move()
        
        # Termination conditions
        asteroid_collision = any(wrapped_distance(a.x, a.y, ship.x, ship.y) < ship.radius + a.radius for a in asteroids)
        out_of_fuel = ship.fuel <= 0
        no_progress = alive_time > 1000 and ship.minerals == 0  # Encourage early mining
        
        if asteroid_collision or out_of_fuel or no_progress or alive_time >= 8000:
            # Penalty for collision or running out of fuel
            if asteroid_collision:
                genome.fitness -= 30
            elif out_of_fuel and ship.minerals == 0:
                genome.fitness -= 15
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
        genome.fitness = 0  # Initialize fitness
        run_simulation(genome, config, visualizer=None)  # No visualization during evaluation
        
        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome
    
    # Update visualizer with this generation's results
    visualizer.update_generation(best_in_generation)
    
    # Visualize the best genome from this generation (every 5th generation)
    if best_in_generation and visualizer.generation % 5 == 0:
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
        winner = population.run(eval_genomes, 200)  # Increased generations
        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")
        
        # Test the winner
        print("\nTesting winner...")
        run_simulation(winner, config, visualizer=config.visualizer)

        # save the winner
        output_dir = "output/winner/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(output_dir, f"winner_{timestamp}.pkl"), "wb") as f:
            pickle.dump(winner, f)
        print(f"Winner genome saved to {output_dir} as 'winner_{timestamp}.pkl'")

    finally:
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)