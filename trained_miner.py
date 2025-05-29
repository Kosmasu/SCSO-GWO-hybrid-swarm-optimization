import pickle
import pygame
import neat
import math
import os
#from miner import Spaceship, Mineral, Asteroid  # Import your game classes random
#from miner_harness import Spaceship, Mineral, Asteroid  # Import your game classes fixed locations
from miner_harness2 import Spaceship, Mineral, Asteroid 
#from miner_harness3 import Spaceship, Mineral, Asteroid 

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_trained_model(filename):
    """
    Load a saved genome and config
    """
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"Model loaded from {filename}")
    return save_data['genome']

def run_with_trained_model():
    # Initialize pygame
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trained Miner Ship")
    clock = pygame.time.Clock()
    
    successful_minings=0

    # Create and store visualizer in config
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Load the trained model
    genome = load_trained_model('trained_miner.pkl')
    
    # Create the neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Game setup
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    font = pygame.font.SysFont(None, 24)
    alive_time=0

    running = True
    while running:
        alive_time+=1
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Find closest objects
        closest_mineral = min((m for m in minerals if m), 
                            key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y))
        closest_asteroid = min(asteroids,
                             key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y))

        # Calculate current distances
        mineral_dist = math.hypot(ship.x-closest_mineral.x, ship.y-closest_mineral.y)
        asteroid_dist = math.hypot(ship.x-closest_asteroid.x, ship.y-closest_asteroid.y)
        
        inputs = [
            #math.hypot(ship.x - closest_mineral.x)/WIDTH if closest_mineral else 0,
            mineral_dist,
            math.atan2(closest_mineral.y-ship.y, closest_mineral.x-ship.x)/math.pi if closest_mineral else 0,
            #math.hypot(ship.x - closest_asteroid.x)/WIDTH if closest_asteroid else 0,
            asteroid_dist,
            #max(min(ship.fuel/100.0, 1.0), 0.0)
            math.atan2(closest_asteroid.y-ship.y, closest_asteroid.x-ship.x)/math.pi if closest_asteroid else 0
        ]
        
        # Get network output
        output = net.activate(inputs)
        
        # Execute actions (same as in training)
        theta_max=math.radians(45)
        ship.angle = output[0]*theta_max
        #ship.angle = angular_velocity * 0.1
        if output[1] >= 0:  # Thrust
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            ship.move(dx, dy)
            
        if output[2] > 0.5:  # Mine
            minerals_to_remove = []
            for mineral in minerals:
                if mineral and math.hypot(ship.x-mineral.x, ship.y-mineral.y) < ship.radius + mineral.radius:
                    minerals_to_remove.append(mineral)
                    successful_minings += 1
                    ship.minerals += 1
                    ship.fuel = min(100, ship.fuel + 15)
            minerals = [m for m in minerals if m not in minerals_to_remove]
            
        
        # Asteroid movement
        for asteroid in asteroids:
            asteroid.move()
            # Collision detection
            dist = math.hypot(ship.x - asteroid.x, ship.y - asteroid.y)
            if dist < ship.radius + asteroid.radius:
                running = False

        # Draw everything
        for mineral in minerals:
            mineral.draw()
        for asteroid in asteroids:
            asteroid.draw()
        ship.draw()

        # Display fuel and minerals
        font = pygame.font.SysFont(None, 36)
        
        # Display stats
        stats = [
            f"Minerals: {ship.minerals}",
            f"Alive Time: {alive_time:0.1f}",
            f"Fuel: {ship.fuel:.1f}",
            f"Score: {ship.minerals*100+alive_time/4:.1f}",
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(30)

    running=True
    while running:
        screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        stats = [
            f"Minerals: {ship.minerals}",
            f"Alive Time: {alive_time:0.1f}",
            f"Fuel: {ship.fuel:.1f}",
            f"Score: {ship.minerals*100+alive_time/4:.1f}",
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 25))
        text = font.render("END", True, (255, 4, 5))
        screen.blit(text, (int(HEIGHT/2), WIDTH/2))

        pygame.display.flip()
        clock.tick(60)   
    pygame.quit()

if __name__ == "__main__":
    run_with_trained_model()