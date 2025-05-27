import pygame
import math
from game import (
    Spaceship,
    Mineral,
    Asteroid,
    get_closest_asteroid_info,
    get_closest_mineral_info,
)
from data import BLACK, GREEN, RED, WHITE, WIDTH, HEIGHT
from miner_neat2 import get_neat_inputs

# Create larger window for debug panel
if not pygame.get_init():
    pygame.init()
screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
clock: pygame.time.Clock = pygame.time.Clock()
WINDOW_WIDTH = WIDTH + 420  # Game width + debug panel width
WINDOW_HEIGHT = HEIGHT
debug_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Space Miner - Debug Mode")

# Debug panel scroll state
debug_scroll_offset = 0
debug_content_height = 0


def draw_debug_panel(
    screen: pygame.Surface,
    ship: Spaceship,
    minerals: list[Mineral],
    asteroids: list[Asteroid],
):
    """Draw a comprehensive debug panel showing all NEAT inputs"""
    global debug_content_height

    inputs_explanation, inputs_value = get_neat_inputs(ship, minerals, asteroids)

    # Create debug surface (make it larger to accommodate all content)
    debug_width = 400
    debug_height = HEIGHT - 20
    max_content_height = 2000  # Large enough for all content
    debug_surface = pygame.Surface((debug_width, max_content_height))
    debug_surface.fill((0, 0, 0))

    font_small = pygame.font.SysFont(None, 20)
    font_medium = pygame.font.SysFont(None, 24)
    y_offset = 10
    line_height = 18

    def draw_text(text, font, color=WHITE, bold=False):
        nonlocal y_offset
        if bold:
            font.set_bold(True)
        rendered = font.render(text, True, color)
        debug_surface.blit(rendered, (10, y_offset))
        y_offset += line_height
        if bold:
            font.set_bold(False)

    # Header
    draw_text("=== NEAT INPUTS DEBUG ===", font_medium, WHITE, True)
    draw_text("Scroll: Mouse Wheel / Page Up/Down", font_small, (200, 200, 200))
    y_offset += 5

    input_idx = 0

    # Proximity Asteroids (3 inputs)
    draw_text("PROXIMITY ASTEROIDS:", font_medium, (255, 255, 0), True)
    for i in range(3):
        explanation = (
            inputs_explanation[input_idx]
            if input_idx < len(inputs_explanation)
            else "Unknown"
        )
        value = inputs_value[input_idx] if input_idx < len(inputs_value) else 0.0
        draw_text(f"  {explanation}: {value:.3f}", font_small)
        input_idx += 1
    y_offset += 5

    # Top 5 Asteroids (8 inputs each)
    draw_text("TOP 5 ASTEROIDS:", font_medium, (255, 100, 100), True)
    for i in range(5):
        draw_text(
            f"  Asteroid {i + 1}:",
            font_small,
            (255, 150, 150),
        )
        # Display 8 inputs for this asteroid
        for j in range(8):
            if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
                explanation = inputs_explanation[input_idx]
                value = inputs_value[input_idx]
                draw_text(f"    {explanation}: {value:.3f}", font_small)
            input_idx += 1

    y_offset += 5

    # Top 5 Minerals (7 inputs each, but only 3 minerals exist)
    draw_text("TOP 5 MINERALS:", font_medium, (100, 255, 100), True)
    for i in range(5):
        draw_text(
            f"  Mineral {i + 1}:",
            font_small,
            (150, 255, 150),
        )
        # Display 7 inputs for existing minerals
        for j in range(7):
            if input_idx < len(inputs_explanation) and input_idx < len(
                inputs_value
            ):
                explanation = inputs_explanation[input_idx]
                value = inputs_value[input_idx]
                draw_text(f"    {explanation}: {value:.3f}", font_small)
            input_idx += 1
    y_offset += 5

    # Ship State (5 inputs)
    draw_text("SHIP STATE:", font_medium, (100, 100, 255), True)
    draw_text(
        f"  Actual values - Angle: {ship.angle:.2f}",
        font_small,
        (150, 150, 255),
    )
    for i in range(5):
        if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
            explanation = inputs_explanation[input_idx]
            value = inputs_value[input_idx]
            draw_text(f"  {explanation}: {value:.3f}", font_small)
        input_idx += 1
    y_offset += 5

    # Spatial Awareness (4 inputs)
    draw_text("SPATIAL AWARENESS:", font_medium, (255, 100, 255), True)
    draw_text(
        f"  Actual position - X: {ship.x:.1f}, Y: {ship.y:.1f}",
        font_small,
        (200, 150, 255),
    )
    for i in range(4):
        if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
            explanation = inputs_explanation[input_idx]
            value = inputs_value[input_idx]
            draw_text(f"  {explanation}: {value:.3f}", font_small)
        input_idx += 1

    draw_text(
        f"TOTAL INPUTS: {len(inputs_value)} (Expected: {input_idx})",
        font_medium,
        (255, 255, 255),
        True,
    )

    # Store content height for scrolling
    debug_content_height = y_offset

    # Create viewport surface
    viewport = pygame.Surface((debug_width, debug_height))
    viewport.fill((0, 0, 0))
    viewport.set_alpha(200)

    # Blit portion of debug surface based on scroll offset
    viewport.blit(debug_surface, (0, -debug_scroll_offset))

    # Draw scrollbar if content is scrollable
    if debug_content_height > debug_height:
        scrollbar_height = max(
            20, int((debug_height / debug_content_height) * debug_height)
        )
        scrollbar_y = int(
            (debug_scroll_offset / (debug_content_height - debug_height))
            * (debug_height - scrollbar_height)
        )
        pygame.draw.rect(
            viewport, (100, 100, 100), (debug_width - 10, 0, 8, debug_height)
        )
        pygame.draw.rect(
            viewport,
            (200, 200, 200),
            (debug_width - 10, scrollbar_y, 8, scrollbar_height),
        )

    # Blit viewport to the right side of the window
    screen.blit(viewport, (WIDTH + 10, 10))


def handle_debug_scroll(event):
    """Handle scrolling events for the debug panel"""
    global debug_scroll_offset

    scroll_speed = 30
    max_scroll = max(0, debug_content_height - (HEIGHT - 20))

    if event.type == pygame.MOUSEWHEEL:
        # Mouse wheel scrolling
        debug_scroll_offset -= event.y * scroll_speed
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_PAGEUP:
            debug_scroll_offset -= scroll_speed * 5
        elif event.key == pygame.K_PAGEDOWN:
            debug_scroll_offset += scroll_speed * 5

    # Clamp scroll offset
    debug_scroll_offset = max(0, min(debug_scroll_offset, max_scroll))


# Game Setup
def main():
    ship = Spaceship(fuel=999999999)
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(1)]
    asteroids[0].speed_x, asteroids[0].speed_y = (
        1.0,
        0.0,
    )
    asteroids[0].x, asteroids[0].y = 0, HEIGHT - 1
    running = True
    debug_mode = False
    alive_time = 0
    dx, dy = 0, 0
    while running:
        alive_time += 1
        debug_screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle debug panel scrolling
            if debug_mode:
                handle_debug_scroll(event)

        # Player controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            ship.angle -= 0.1
            ship.angle = ship.angle % (2 * math.pi)  # Clamp to 0-2π
        if keys[pygame.K_RIGHT]:
            ship.angle += 0.1
            ship.angle = ship.angle % (2 * math.pi)  # Clamp to 0-2π
        if keys[pygame.K_UP]:
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            ship.move(dx, dy)
        if keys[pygame.K_BACKSPACE]:
            debug_mode = not debug_mode
        if keys[pygame.K_w]:
            ship.angle = 3 * math.pi / 2  # Inverted Y-axis for upward movement
            print("Up pressed")
            print(f"Ship angle: {ship.angle}")
            print(f"Ship math.sin(angle): {math.sin(ship.angle)}")
            print(f"Ship math.cos(angle): {math.cos(ship.angle)}")
        if keys[pygame.K_s]:
            ship.angle = math.pi / 2  # Inverted Y-axis for downward movement
            print("Down pressed")
            print(f"Ship angle: {ship.angle}")
            print(f"Ship math.sin(angle): {math.sin(ship.angle)}")
            print(f"Ship math.cos(angle): {math.cos(ship.angle)}")
        if keys[pygame.K_a]:
            ship.angle = math.pi  # Looking left
            print("Left pressed")
            print(f"Ship angle: {ship.angle}")
            print(f"Ship math.sin(angle): {math.sin(ship.angle)}")
            print(f"Ship math.cos(angle): {math.cos(ship.angle)}")
        if keys[pygame.K_d]:
            ship.angle = 0  # Looking right
            print("Right pressed")
            print(f"Ship angle: {ship.angle}")
            print(f"Ship math.sin(angle): {math.sin(ship.angle)}")
            print(f"Ship math.cos(angle): {math.cos(ship.angle)}")

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
            mineral.draw(screen)
        for asteroid in asteroids:
            asteroid.draw(screen)

        if debug_mode:
            # Draw circle around the ship
            pygame.draw.circle(debug_screen, WHITE, (int(ship.x), int(ship.y)), 50, 1)
            pygame.draw.circle(debug_screen, WHITE, (int(ship.x), int(ship.y)), 100, 1)
            pygame.draw.circle(debug_screen, WHITE, (int(ship.x), int(ship.y)), 150, 1)

            # Draw line from ship using the relative angle to minerals
            mineral_info = get_closest_mineral_info(ship, minerals, top_n=3)

            # Draw ship's current facing direction (red line)
            ship_end_x = ship.x + 80 * math.cos(ship.angle)
            ship_end_y = ship.y + 80 * math.sin(ship.angle)
            pygame.draw.line(
                debug_screen,
                (255, 0, 0),
                (int(ship.x), int(ship.y)),
                (int(ship_end_x), int(ship_end_y)),
                3,
            )

            # Draw lines to minerals with relative angles
            for i, info in enumerate(mineral_info):
                # Use the actual mineral position for direct line (wrapped if needed)
                mineral = info.mineral

                # Calculate wrapped distance for visualization
                dx = mineral.x - ship.x
                dy = mineral.y - ship.y

                # Handle screen wrapping for visualization
                if dx > WIDTH / 2:
                    dx -= WIDTH
                elif dx < -WIDTH / 2:
                    dx += WIDTH

                if dy > HEIGHT / 2:
                    dy -= HEIGHT
                elif dy < -HEIGHT / 2:
                    dy += HEIGHT

                # Calculate the wrapped angle for visualization
                wrapped_angle = math.atan2(dy, dx)

                # Draw line showing the relative angle (what the AI sees)
                rel_end_x = ship.x + 80 * math.cos(ship.angle + info.relative_angle)
                rel_end_y = ship.y + 80 * math.sin(ship.angle + info.relative_angle)
                pygame.draw.line(
                    debug_screen,
                    RED,
                    (int(ship.x), int(ship.y)),
                    (int(rel_end_x), int(rel_end_y)),
                    2,
                )

                # Draw line using wrapped angle (should match the relative angle line)
                wrapped_end_x = ship.x + 80 * math.cos(wrapped_angle)
                wrapped_end_y = ship.y + 80 * math.sin(wrapped_angle)
                pygame.draw.line(
                    debug_screen,
                    GREEN,
                    (int(ship.x), int(ship.y)),
                    (int(wrapped_end_x), int(wrapped_end_y)),
                    1,
                )

            # Draw lines to asteroids with relative angles
            asteroid_info = get_closest_asteroid_info(ship, asteroids, top_n=5)
            for i, info in enumerate(asteroid_info):
                # info.angle is already the relative angle from get_closest_asteroid_info
                # So we need to add it to ship.angle to get the absolute direction

                # Draw line showing the relative angle (what the AI sees)
                rel_end_x = ship.x + 80 * math.cos(ship.angle + info.relative_angle)
                rel_end_y = ship.y + 80 * math.sin(ship.angle + info.relative_angle)
                pygame.draw.line(
                    debug_screen,
                    RED,
                    (int(ship.x), int(ship.y)),
                    (int(rel_end_x), int(rel_end_y)),
                    2,
                )

                # Draw line directly to the asteroid (absolute direction for reference)
                asteroid = info.asteroid
                pygame.draw.line(
                    debug_screen,
                    WHITE,
                    (int(ship.x), int(ship.y)),
                    (int(asteroid.x), int(asteroid.y)),
                    1,
                )

                # Add text labels
                font_small = pygame.font.SysFont(None, 20)
                rel_text = font_small.render(
                    f"Rel: {math.degrees(info.relative_angle):.1f}°",
                    True,
                    (255, 255, 255),
                )
                debug_screen.blit(rel_text, (int(rel_end_x + 5), int(rel_end_y)))

            # Draw comprehensive debug panel
            draw_debug_panel(debug_screen, ship, minerals, asteroids)

            # Draw a vertical line to separate game area from debug panel
            pygame.draw.line(debug_screen, WHITE, (WIDTH, 0), (WIDTH, HEIGHT), 2)

        ship.draw(screen)

        # Display fuel and minerals
        font = pygame.font.SysFont(None, 36)
        fuel_text = font.render(f"Fuel: {ship.fuel:.1f}", True, WHITE)
        minerals_text = font.render(f"Minerals: {ship.minerals}", True, WHITE)
        alive_time_text = font.render(f"Alive Time: {alive_time}", True, WHITE)
        debug_text = font.render(
            f"Debug Mode: {'ON' if debug_mode else 'OFF'} (Backspace to toggle)",
            True,
            WHITE,
        )
        debug_screen.blit(fuel_text, (10, 10))
        debug_screen.blit(minerals_text, (10, 50))
        debug_screen.blit(alive_time_text, (10, 90))
        debug_screen.blit(debug_text, (10, 130))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
