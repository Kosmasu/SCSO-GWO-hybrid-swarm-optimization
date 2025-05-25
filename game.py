
from pydantic import BaseModel, Field
from data import WIDTH, HEIGHT, WHITE, BLUE, YELLOW, RED
import random
import math

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