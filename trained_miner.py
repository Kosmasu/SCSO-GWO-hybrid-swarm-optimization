import pickle
import pygame
import neat
import math
import os
#from miner import Spaceship, Mineral, Asteroid  # Import your game classes random
# from miner_harness import Spaceship, Mineral, Asteroid  # Import your game classes fixed locations
from miner_harness2 import Spaceship, Mineral, Asteroid
# from miner_harness3 import Spaceship, Mineral, Asteroid 
# from miner_neat2 import get_neat_inputs 
from miner_neat2_both import get_neat_inputs 
# from miner_neat2_radar import get_neat_inputs 
# from miner_neat2_asteroid import get_neat_inputs 
# from miner_neat2_asteroid import get_neat_inputs 

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_trained_model(filename):
    """
    Load a saved genome and config
    """
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"Model loaded from {filename}")
    return save_data

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
    genome = load_trained_model('output/neat/20250604-211551/best_genome_gen_278_fitness_14570.4.pkl')
    
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

        _, inputs = get_neat_inputs(ship, minerals, asteroids)
        
        # Get network output
        output = net.activate(inputs)
        
        # Execute actions (same as in training)
        turn_output = output[0]
        if turn_output < -0.3:
            turn_rate = ((turn_output + 0.3) / 0.7) * 0.15
        elif turn_output >= -0.3 and turn_output <= 0.3:
            turn_rate = 0
        else:
            turn_rate = ((turn_output - 0.3) / 0.7) * 0.15
        ship.angle += turn_rate
        ship.angle = ship.angle % (2 * math.pi)
        #ship.angle = angular_velocity * 0.1
        thrust_output = output[1]
        if thrust_output < -0.3:
            thrust_power = ((thrust_output + 0.3) / 0.7) * 0.8
        elif thrust_output >= -0.3 and thrust_output <= 0.3:
            thrust_power = 0
        else:
            thrust_power = (thrust_output - 0.3) / 0.7

        dx = thrust_power * ship.speed * math.cos(ship.angle)
        dy = thrust_power * ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)
            
        minerals_to_remove = []
        for mineral in minerals:
            if mineral and math.hypot(ship.x-mineral.x, ship.y-mineral.y) < ship.radius + mineral.radius:
                minerals_to_remove.append(mineral)
                successful_minings += 1
                ship.minerals += 1
                ship.fuel = min(100, ship.fuel + 15)
        minerals = [m for m in minerals if m not in minerals_to_remove]
        if len(minerals) < 3:  # Spawn new minerals if too few
            minerals.extend(Mineral() for _ in range(2))
            
        
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
        clock.tick(60)

if __name__ == "__main__":
    run_with_trained_model()