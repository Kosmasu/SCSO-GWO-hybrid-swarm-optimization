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
            # Store the actual velocity for AI input calculation
            self.velocity_x = dx
            self.velocity_y = dy
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            if dx != 0 or dy != 0:
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

    def move(self, speed_multiplier: float = 1.0) -> None:
        self.x = (self.x + self.speed_x * speed_multiplier) % WIDTH
        self.y = (self.y + self.speed_y * speed_multiplier) % HEIGHT

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
            )
        )

    # Sort by surface distance and take top N
    mineral_data.sort(key=lambda x: x.distance)
    top_minerals = mineral_data[:top_n]

    return top_minerals


class AsteroidInfo(BaseModel):
    distance: float
    relative_angle: float
    velocity_angle: float  # Add this field
    velocity_magnitude: float  # Add this field for completeness
    future_positions: list[tuple[float, float]]  # [(distance, angle), ...]
    radius: float


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

        # Calculate asteroid velocity angle and magnitude
        velocity_magnitude = math.sqrt(asteroid.speed_x**2 + asteroid.speed_y**2)
        velocity_angle_absolute = math.atan2(asteroid.speed_y, asteroid.speed_x)

        # Convert to relative angle (relative to ship's heading)
        velocity_angle_relative = velocity_angle_absolute - ship.angle
        while velocity_angle_relative > math.pi:
            velocity_angle_relative -= 2 * math.pi
        while velocity_angle_relative < -math.pi:
            velocity_angle_relative += 2 * math.pi

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
                velocity_angle=velocity_angle_relative,
                velocity_magnitude=velocity_magnitude,
                future_positions=future_positions,
                radius=asteroid.radius,
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
    objects: list[Mineral | Asteroid],
    n_directions: int = 24,
    max_range: float = 300.0,
) -> list[RadarScanResult]:
    results = []
    angle_step = 2 * math.pi / n_directions  # Use radians for consistency

    for i in range(n_directions):
        # Calculate relative angle from ship's heading
        relative_angle_rad = (i * angle_step) - math.pi  # Range: -π to π

        # Calculate absolute direction based on ship's heading + relative angle
        absolute_angle_rad = ship.angle + relative_angle_rad

        dx = math.cos(absolute_angle_rad)
        dy = math.sin(absolute_angle_rad)

        closest_dist = max_range

        for obj in objects:
            # Use wrapped distance and direction
            _, ox, oy = calculate_wrapped_distance(ship.x, ship.y, obj.x, obj.y)

            # Project object vector onto radar direction
            proj_len = ox * dx + oy * dy  # scalar projection

            if proj_len <= 0 or proj_len > max_range:
                continue  # object is behind or too far

            # Closest point on the radar line to the object's center
            closest_x = ox - dx * proj_len
            closest_y = oy - dy * proj_len

            # Distance from object center to radar line
            dist_to_line = math.hypot(ox - dx * proj_len, oy - dy * proj_len)

            total_radius = ship.radius + obj.radius
            if dist_to_line <= total_radius:
                # Use Pythagoras to compute actual hit distance from ship to surface
                offset = math.sqrt(max(total_radius**2 - dist_to_line**2, 0))
                hit_distance = proj_len - offset

                if 0 < hit_distance < closest_dist:
                    closest_dist = hit_distance

        # Store relative angle in radians
        results.append(RadarScanResult(angle=relative_angle_rad, distance=closest_dist))

    return results

class MineralRiskInfo(BaseModel):
    mineral: Mineral
    distance: float
    relative_angle: float
    risk_score: float
    risk_factors: dict[str, float]
    normalized_risk: float  # Add this field


def calculate_mineral_risk(
    ship: Spaceship,
    mineral: Mineral,
    asteroids: list[Asteroid],
    future_frames: list[int] = [30, 60, 90],
) -> tuple[float, dict[str, float]]:
    """
    Calculate risk score for mining a specific mineral.
    Lower score = safer to mine
    Higher score = more dangerous
    """
    risk_factors = {}

    # 1. Current collision risk - Is mineral currently touching an asteroid?
    current_collision_risk = 0
    for asteroid in asteroids:
        distance, _, _ = calculate_wrapped_distance(
            mineral.x, mineral.y, asteroid.x, asteroid.y
        )
        if distance <= mineral.radius + asteroid.radius + 5:
            current_collision_risk = 1000  # Extremely high risk
            break
    risk_factors["current_collision"] = current_collision_risk

    # 2. Proximity risk - Count asteroids near the mineral
    proximity_risk = 0
    danger_radius = 60
    nearby_asteroids = count_asteroids_in_radius(mineral, asteroids, danger_radius)
    proximity_risk = len(nearby_asteroids) * 50
    risk_factors["proximity"] = proximity_risk

    # 3. Path obstruction risk
    path_obstruction_risk = 0
    ship_to_mineral_distance, dx, dy = calculate_wrapped_distance(
        ship.x, ship.y, mineral.x, mineral.y
    )

    if ship_to_mineral_distance > 0:
        path_dx = dx / ship_to_mineral_distance
        path_dy = dy / ship_to_mineral_distance

        for asteroid in asteroids:
            asteroid_distance, asteroid_dx, asteroid_dy = calculate_wrapped_distance(
                ship.x, ship.y, asteroid.x, asteroid.y
            )

            if asteroid_distance > 0 and asteroid_distance < ship_to_mineral_distance:
                asteroid_dir_x = asteroid_dx / asteroid_distance
                asteroid_dir_y = asteroid_dy / asteroid_distance

                dot_product = path_dx * asteroid_dir_x + path_dy * asteroid_dir_y

                if dot_product > 0.8:
                    cross_product = abs(path_dx * asteroid_dy - path_dy * asteroid_dx)
                    path_distance = cross_product * asteroid_distance

                    if path_distance < asteroid.radius + ship.radius + 20:
                        path_obstruction_risk += 100

    risk_factors["path_obstruction"] = path_obstruction_risk

    # 4. Future collision risk
    future_collision_risk = 0
    for frames in future_frames:
        for asteroid in asteroids:
            future_x = (asteroid.x + asteroid.speed_x * frames) % WIDTH
            future_y = (asteroid.y + asteroid.speed_y * frames) % HEIGHT

            future_distance, _, _ = calculate_wrapped_distance(
                mineral.x, mineral.y, future_x, future_y
            )

            if future_distance <= mineral.radius + asteroid.radius + 10:
                time_weight = 1.0 / (frames / 30.0)
                future_collision_risk += 200 * time_weight

    risk_factors["future_collision"] = future_collision_risk

    # 5. Escape route risk
    escape_risk = 0
    escape_directions = 8
    blocked_directions = 0

    for i in range(escape_directions):
        escape_angle = (2 * math.pi * i) / escape_directions
        escape_distance = 50

        escape_x = mineral.x + escape_distance * math.cos(escape_angle)
        escape_y = mineral.y + escape_distance * math.sin(escape_angle)

        escape_x = escape_x % WIDTH
        escape_y = escape_y % HEIGHT

        for asteroid in asteroids:
            distance, _, _ = calculate_wrapped_distance(
                escape_x, escape_y, asteroid.x, asteroid.y
            )
            if distance <= asteroid.radius + ship.radius + 15:
                blocked_directions += 1
                break

    escape_risk = (blocked_directions / escape_directions) * 150
    risk_factors["escape_route"] = escape_risk

    # 6. Distance penalty
    distance_risk = ship_to_mineral_distance * 0.5
    risk_factors["distance"] = distance_risk

    # 7. Asteroid velocity risk
    velocity_risk = 0
    for asteroid in nearby_asteroids:
        asteroid_speed = math.sqrt(asteroid.speed_x**2 + asteroid.speed_y**2)
        velocity_risk += asteroid_speed * 10
    risk_factors["velocity"] = velocity_risk

    total_risk = sum(risk_factors.values())

    return total_risk, risk_factors


def get_safest_minerals(
    ship: Spaceship,
    minerals: list[Mineral],
    asteroids: list[Asteroid],
    max_risk_threshold: float = None,  # Remove fixed threshold
    top_n: int = 3,
) -> list[MineralRiskInfo]:
    """
    Get the safest minerals to mine, sorted by risk score with proper normalization.
    """
    if not minerals:
        return []

    mineral_risks = []
    risk_scores = []  # Collect all risk scores for normalization

    # First pass: calculate all risk scores
    for mineral in minerals:
        distance, _, _ = calculate_wrapped_distance(
            ship.x, ship.y, mineral.x, mineral.y
        )
        relative_angle = calculate_relative_angle(
            ship.angle, ship.x, ship.y, mineral.x, mineral.y
        )

        risk_score, risk_factors = calculate_mineral_risk(ship, mineral, asteroids)
        risk_scores.append(risk_score)

        mineral_risks.append(
            {
                "mineral": mineral,
                "distance": distance,
                "relative_angle": relative_angle,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
            }
        )

    # Calculate normalization parameters
    if risk_scores:
        min_risk = min(risk_scores)
        max_risk = max(risk_scores)
        risk_range = max_risk - min_risk if max_risk > min_risk else 1.0
    else:
        min_risk = max_risk = risk_range = 0

    # Second pass: create normalized MineralRiskInfo objects
    normalized_mineral_risks = []
    for mineral_data in mineral_risks:
        # Normalize risk score to [0, 1] where 0 = highest risk, 1 = lowest risk
        if risk_range > 0:
            normalized_risk = 1.0 - (
                (mineral_data["risk_score"] - min_risk) / risk_range
            )
        else:
            normalized_risk = 1.0  # All minerals have same risk

        # Apply threshold filtering if specified
        if (
            max_risk_threshold is not None
            and mineral_data["risk_score"] > max_risk_threshold
        ):
            continue

        normalized_mineral_risks.append(
            MineralRiskInfo(
                mineral=mineral_data["mineral"],
                distance=mineral_data["distance"],
                relative_angle=mineral_data["relative_angle"],
                risk_score=mineral_data["risk_score"],
                risk_factors=mineral_data["risk_factors"],
                normalized_risk=normalized_risk,
            )
        )

    # Sort by risk score (lowest first = safest first)
    normalized_mineral_risks.sort(key=lambda x: x.risk_score)

    return normalized_mineral_risks[:top_n]
