from pydantic import BaseModel, Field
import pygame
from data import (
    ASTEROID_MAX_RADIUS,
    ASTEROID_MAX_SPEED,
    ASTEROID_MIN_RADIUS,
    ASTEROID_MIN_SPEED,
    MINERAL_RADIUS,
    WIDTH,
    HEIGHT,
    WHITE,
    BLUE,
    YELLOW,
    RED,
)
import random
import math


# Game Classes (modified to use Pydantic BaseModel)
# Original class order is maintained.
class Spaceship(BaseModel):
    x: float = Field(default_factory=lambda: float(WIDTH // 2))
    y: float = Field(default_factory=lambda: float(HEIGHT // 2))
    speed: float = 5.0
    angle: float = 0.0
    fuel: float = 100.0
    minerals: int = 0
    radius: int = 15
    # Add velocity tracking (for AI input only, doesn't affect actual movement)
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def move(self, dx: float, dy: float) -> None:
        if self.fuel > 0:
            # Store the velocity for AI input calculation
            self.velocity_x = dx
            self.velocity_y = dy
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.fuel -= 0.1  # fuel is float
        else:
            self.velocity_x = 0
            self.velocity_y = 0

    def mine(self, minerals: list["Mineral"]) -> None:
        for mineral_obj in minerals[:]:
            dist: float = math.hypot(self.x - mineral_obj.x, self.y - mineral_obj.y)
            if dist < self.radius + mineral_obj.radius:
                minerals.remove(mineral_obj)
                self.minerals += 1
                self.fuel = min(100.0, self.fuel + 10.0)

    def draw(self, screen) -> None:
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
            (right_fin_x, right_fin_y),
        ]
        pygame.draw.polygon(screen, WHITE, points)

    def debug(self) -> None:
        print(
            f"""
Spaceship
Position: ({self.x:.2f}, {self.y:.2f})
Speed: {self.speed}
Angle: {math.degrees(self.angle):.2f} degrees
Fuel: {self.fuel:.2f}
Minerals: {self.minerals}
Radius: {self.radius}
===========================
        """.strip()
        )

    class Config:
        arbitrary_types_allowed = (
            True  # Useful for Pydantic models with complex field types
        )


class Mineral(BaseModel):
    x: int = Field(default_factory=lambda: random.randint(20, WIDTH - 20))
    y: int = Field(default_factory=lambda: random.randint(20, HEIGHT - 20))
    radius: int = MINERAL_RADIUS

    def draw(self, screen) -> None:
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.radius)

    class Config:
        arbitrary_types_allowed = True

    def debug(self) -> None:
        print(
            f"""
Mineral 
Position: ({self.x}, {self.y})
Radius: {self.radius}
=========================== 
        """.strip()
        )


class Asteroid(BaseModel):
    x: float = Field(default_factory=lambda: float(random.randint(0, WIDTH)))
    y: float = Field(default_factory=lambda: float(random.randint(0, HEIGHT)))
    radius: int = Field(
        default_factory=lambda: random.randint(ASTEROID_MIN_RADIUS, ASTEROID_MAX_RADIUS)
    )
    speed_x: float = Field(
        default_factory=lambda: random.uniform(ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED)
    )
    speed_y: float = Field(
        default_factory=lambda: random.uniform(ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED)
    )

    def move(self) -> None:
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self, screen) -> None:
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

    def debug(self) -> None:
        print(
            f"""
Asteroid :
Position: ({self.x:.2f}, {self.y:.2f})
Speed: ({self.speed_x:.2f}, {self.speed_y:.2f})
Radius: {self.radius}
Future Position: ({(self.x + self.speed_x) % WIDTH:.2f}, {(self.y + self.speed_y) % HEIGHT:.2f})
===========================
        """.strip()
        )

    class Config:
        arbitrary_types_allowed = True


def calculate_wrapped_distance(
    x1: float, y1: float, x2: float, y2: float, width: int = WIDTH, height: int = HEIGHT
):
    """
    Calculate the shortest distance between two points considering screen wrapping.

    Args:
        x1, y1: Position of first object (e.g., spaceship)
        x2, y2: Position of second object (e.g., asteroid/mineral)
        width, height: Screen dimensions

    Returns:
        tuple: (distance, dx, dy) where dx and dy are the shortest direction vectors
    """
    # Calculate direct distance
    dx_direct = x2 - x1
    dy_direct = y2 - y1

    # Calculate wrapped distances
    dx_wrap = (
        dx_direct - width
        if dx_direct > width / 2
        else dx_direct + width
        if dx_direct < -width / 2
        else dx_direct
    )
    dy_wrap = (
        dy_direct - height
        if dy_direct > height / 2
        else dy_direct + height
        if dy_direct < -height / 2
        else dy_direct
    )

    # Return the shortest distance and direction
    distance = math.sqrt(dx_wrap**2 + dy_wrap**2)
    return distance, dx_wrap, dy_wrap


def calculate_relative_angle(
    angle: float,
    x_1: float,
    y_1: float,
    x_2: float,
    y_2: float,
):
    """
    Calculate the relative angle between the ship's current facing direction and the direction to a point.
    Args:
        angle: Current angle of the ship in radians
        x_1, y_1: Coordinates of the ship
        x_2, y_2: Coordinates of the point (e.g., mineral or asteroid)
    Returns:
        float: Relative angle in radians, normalized to [-π, π]
    """
    dx = x_2 - x_1
    dy = y_2 - y_1
    absolute_angle = math.atan2(dy, dx)

    # Normalize to [0, 2π]
    absolute_angle = (
        absolute_angle if absolute_angle >= 0 else absolute_angle + 2 * math.pi
    )

    # Calculate relative angle
    relative_angle = absolute_angle - angle
    while relative_angle > math.pi:
        relative_angle -= 2 * math.pi
    while relative_angle < -math.pi:
        relative_angle += 2 * math.pi

    return relative_angle


class MineralInfo(BaseModel):
    distance: float
    relative_angle: float
    mineral: Mineral


def get_closest_mineral_info(
    ship: Spaceship, minerals: list[Mineral], top_n: int = 1
) -> list[MineralInfo]:
    """
    Get information about the top N closest minerals considering wrapping and surface-to-surface distance.

    Args:
        ship: Spaceship object
        minerals: List of Mineral objects
        top_n: Number of closest minerals to return

    Returns:
        list[MineralInfo]: List of MineralInfo objects sorted by surface distance
    """
    if not minerals:
        return []

    mineral_data = []

    for mineral in minerals:
        center_distance, dx, dy = calculate_wrapped_distance(
            ship.x, ship.y, mineral.x, mineral.y
        )

        # Calculate surface-to-surface distance
        surface_distance = max(0, center_distance - ship.radius - mineral.radius)

        relative_angle = calculate_relative_angle(
            ship.angle, ship.x, ship.y, mineral.x, mineral.y
        )

        mineral_data.append(
            MineralInfo(
                distance=surface_distance,
                relative_angle=relative_angle,
                mineral=mineral,
            )
        )

    # Sort by surface distance and take top N
    mineral_data.sort(key=lambda x: x.distance)
    top_minerals = mineral_data[:top_n]

    return top_minerals


class AsteroidInfo(BaseModel):
    distance: float
    relative_angle: float
    future_positions: list[tuple[float, float]]  # [(distance, angle), ...]
    asteroid: Asteroid


def get_closest_asteroid_info(
    ship: Spaceship,
    asteroids: list[Asteroid],
    top_n: int = 1,
    future_frames: list[int] = [30, 60, 90, 120],
) -> list[AsteroidInfo]:
    """
    Get information about the closest asteroid considering wrapping and future positions.

    Args:
        ship: Spaceship object
        asteroids: List of Asteroid objects
        top_n: Number of closest asteroids to return
        future_frames: List of frame counts to predict future positions for

    Returns:
        list[AsteroidInfo]: List of asteroid info objects with future predictions
    """
    if not asteroids:
        return []

    asteroid_data = []

    for asteroid in asteroids:
        # Current position distance
        center_distance, dx, dy = calculate_wrapped_distance(
            ship.x, ship.y, asteroid.x, asteroid.y
        )

        # Calculate surface-to-surface distance (0 means touching/collision)
        surface_distance = max(0, center_distance - ship.radius - asteroid.radius)

        relative_angle = calculate_relative_angle(
            ship.angle, ship.x, ship.y, asteroid.x, asteroid.y
        )

        # Calculate future positions at multiple time steps
        future_positions = []
        for frames in future_frames:
            future_x = (asteroid.x + asteroid.speed_x * frames) % WIDTH
            future_y = (asteroid.y + asteroid.speed_y * frames) % HEIGHT
            future_center_distance, future_dx, future_dy = calculate_wrapped_distance(
                ship.x, ship.y, future_x, future_y
            )

            # Calculate future surface-to-surface distance
            future_surface_distance = max(
                0, future_center_distance - ship.radius - asteroid.radius
            )

            future_relative_angle = calculate_relative_angle(
                ship.angle, ship.x, ship.y, future_x, future_y
            )

            future_positions.append((future_surface_distance, future_relative_angle))

        asteroid_data.append(
            AsteroidInfo(
                distance=surface_distance,
                relative_angle=relative_angle,
                future_positions=future_positions,
                asteroid=asteroid,
            )
        )

    # Sort by distance and take top N
    asteroid_data.sort(key=lambda x: x.distance)
    top_asteroids = asteroid_data[:top_n]

    return top_asteroids


def count_asteroids_in_radius(
    object: Spaceship | Mineral, asteroids: list[Asteroid], radius: float
) -> list[Asteroid]:
    """
    Count the number of asteroids within a given radius from the ship's position.
    Args:
        ship: Spaceship object
        asteroids: List of Asteroid objects
        radius: Radius to check for asteroids
    Returns:
        list[Asteroid]: List of asteroids within the specified radius
    """
    proximity_asteroids: list[Asteroid] = []
    for asteroid in asteroids:
        distance, _, _ = calculate_wrapped_distance(
            object.x, object.y, asteroid.x, asteroid.y
        )
        if distance <= radius + asteroid.radius:
            proximity_asteroids.append(asteroid)
    return proximity_asteroids

class RadarScanResult(BaseModel):
    distance: float
    angle: float

def radar_scan(
    ship: Spaceship,
    objects: list[Asteroid] | list[Mineral],
    n_directions: int = 24,
    max_range: float = 300.0,
) -> list[RadarScanResult]:

    radar_results: list[RadarScanResult] = []

    # Calculate angle step for each direction
    angle_step = 2 * math.pi / n_directions

    for i in range(n_directions):
        # Calculate radar beam angle RELATIVE to ship's facing direction
        relative_radar_angle = i * angle_step
        absolute_radar_angle = ship.angle + relative_radar_angle

        # Direction vector for this radar beam (in absolute coordinates)
        beam_dx = math.cos(absolute_radar_angle)
        beam_dy = math.sin(absolute_radar_angle)

        closest_distance = max_range

        # Check each asteroid
        for asteroid in objects:
            # Get vector from ship to asteroid (considering wrapping)
            distance, dx, dy = calculate_wrapped_distance(
                ship.x, ship.y, asteroid.x, asteroid.y
            )

            # Skip if asteroid is beyond max range
            if distance > max_range:
                continue

            # Normalize the direction vector to asteroid
            if distance > 0:
                asteroid_dx = dx / distance
                asteroid_dy = dy / distance

                # Calculate dot product to see if asteroid is in this radar direction
                dot_product = beam_dx * asteroid_dx + beam_dy * asteroid_dy

                # Calculate the angular tolerance (half the angle between adjacent beams)
                angular_tolerance = math.cos(angle_step / 2)

                # If asteroid is within the radar beam cone
                if dot_product >= angular_tolerance:
                    # Calculate surface-to-surface distance (distance to edge of asteroid)
                    surface_distance = max(0, distance - ship.radius - asteroid.radius)
                    
                    # Keep track of the closest obstacle in this direction
                    closest_distance = min(closest_distance, surface_distance)

        radar_results.append(
            RadarScanResult(
                distance=closest_distance,
                angle=relative_radar_angle,  # Store relative angle
            )
        )

    return radar_results
