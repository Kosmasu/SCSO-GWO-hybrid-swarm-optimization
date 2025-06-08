# harness 2

import pygame
import random
import math

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Miner")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Player (Spaceship)
class Spaceship:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.fuel = 100
        self.minerals = 0
        self.radius = 15

    def move(self, dx, dy):
        if self.fuel > 0:
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.fuel -= 0.1

    def mine(self, minerals):
        for mineral in minerals[:]:
            dist = math.hypot(self.x - mineral.x, self.y - mineral.y)
            if dist < self.radius + mineral.radius:
                minerals.remove(mineral)
                self.minerals += 1
                self.fuel = min(100, self.fuel + 10)  # Refuel when mining

    def draw(self, screen) -> None:
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)
        # Draw a triangle for direction
        points = [
            (self.x + self.radius * math.cos(self.angle), 
             self.y + self.radius * math.sin(self.angle)),
            (self.x + self.radius * math.cos(self.angle + 2.5), 
             self.y + self.radius * math.sin(self.angle + 2.5)),
            (self.x + self.radius * math.cos(self.angle - 2.5), 
             self.y + self.radius * math.sin(self.angle - 2.5))
        ]
        pygame.draw.polygon(screen, WHITE, points)

# Mineral
class Mineral:
    _coordinates = [(323,550), (526,229), (541,337), (487,491), (725,214), (478,131), (437,269), (598,41), (666,440), (103,109)]
    _index = 0

    @staticmethod
    def get_next_coord():
        coord = Mineral._coordinates[Mineral._index]
        Mineral._index = (Mineral._index + 1) % len(Mineral._coordinates)  # Cycle through
        return coord

    def __init__(self):
        #self.x = random.randint(20, WIDTH - 20)
        #self.y = random.randint(20, HEIGHT - 20)
        self.x, self.y = Mineral.get_next_coord()  # Get next fixed coordinate
        self.radius = 10

    def draw(self, screen) -> None:
        pygame.draw.circle(screen, BLUE, (self.x, self.y), self.radius)

# Asteroids (Obstacles)
class Asteroid:
    _coordinates = [(381,10), (799,486), (618,91), (324,355), (699,529), (194,355), (372,21), (300,415), (123,472), (800,461),]
    _radius = [25, 27, 18, 23, 15, 19, 21, 17, 17, 23,]
    _speed = [(0.011707815121516862,1.3009039995955534), (-0.8360569150346664,1.1184186184932683), (1.892273483620047,-0.30748917431982337), (
        -0.2045859043086793,-1.0879981057597452), (-0.6505109586084119,-1.2169926405528262), (-0.0065108885771079095,1.6028364789854512), 
        (-0.06388823253179643,-0.1554855882823012), (-0.5838277682211102,1.5174094167227707), (-1.9787026302011501,0.5349021471720463), 
        (-0.11228617731287827,-1.1934599593062036),]
    _index = 0

    @staticmethod
    def get_next_coord():
        coord = Asteroid._coordinates[Asteroid._index]
        radius = Asteroid._radius[Asteroid._index]
        speed = Asteroid._speed[Asteroid._index]
        Asteroid._index = (Asteroid._index + 1) % len(Asteroid._coordinates)  # Cycle through
        return coord, radius, speed
    
    def __init__(self):
        #self.x = random.randint(0, WIDTH)
        #self.y = random.randint(0, HEIGHT)
        #self.radius = random.randint(15, 30)
        #self.speed_x = random.uniform(-2, 2)
        #self.speed_y = random.uniform(-2, 2)
        (self.x, self.y), self.radius, (self.speed_x, self.speed_y) = Asteroid.get_next_coord()  # Get next fixed coordinate

    def move(self):
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self, screen) -> None:
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

# Game Setup
def main():
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    running = True
    score = 0
    alive_time = 0

    while running:
        alive_time+=1
        screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Player controls
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:
            ship.angle -= 0.1
        if keys[pygame.K_RIGHT]:
            ship.angle += 0.1
        if keys[pygame.K_UP]:
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)

        # Mining
        if keys[pygame.K_SPACE]:
            ship.mine(minerals)
            if len(minerals) < 3:  # Spawn new minerals if too few
                minerals.append(Mineral())

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
        alive_text = font.render(f"Alive: {alive_time:.1f}", True, WHITE)
        fuel_text = font.render(f"Fuel: {ship.fuel:.1f}", True, WHITE)
        minerals_text = font.render(f"Minerals: {ship.minerals*100}", True, WHITE)
        score = alive_time/4+ship.minerals*100
        score_text = font.render(f"SCORE: {score:.1f}", True, WHITE)
        screen.blit(fuel_text, (10, 10))
        screen.blit(alive_text, (10, 50))
        screen.blit(minerals_text, (10, 90))
        screen.blit(score_text, (10, 130))

        pygame.display.flip()
        clock.tick(60)

    running=True
    while running:
        screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.blit(fuel_text, (300, 110))
        screen.blit(alive_text, (300, 150))
        screen.blit(minerals_text, (300, 190))
        screen.blit(score_text, (300, 230))
        pygame.display.flip()
        clock.tick(60)
    

    pygame.quit()

if __name__ == "__main__":
    main()