import pygame
import math
from game import (
    RadarScanResult,
    Spaceship,
    Mineral,
    Asteroid,
    get_closest_asteroid_info,
    get_closest_mineral_info,
    radar_scan,
)
from data import BLACK, GREEN, RED, WHITE, WIDTH, HEIGHT
from miner_neat2 import get_neat_inputs

# Create larger window for debug panel
if not pygame.get_init():
    pygame.init()
screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
clock: pygame.time.Clock = pygame.time.Clock()
WINDOW_WIDTH = WIDTH + 620  # Game width + debug panel width
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
    debug_width = 600
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

    # Ship State (3 inputs)
    draw_text("SHIP STATE:", font_medium, (100, 100, 255), True)
    draw_text(
        f"  Actual values - Angle: {ship.angle:.2f}",
        font_small,
        (150, 150, 255),
    )
    for i in range(3):
        if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
            explanation = inputs_explanation[input_idx]
            value = inputs_value[input_idx]
            draw_text(f"  {explanation}: {value:.3f}", font_small)
        input_idx += 1
    y_offset += 5

    # Asteroid Radar Scan (12 inputs)
    draw_text("ASTEROID RADAR SCAN:", font_medium, (255, 255, 100), True)
    draw_text(
        f"  12 directions, 200px max range",
        font_small,
        (255, 255, 150),
    )
    for i in range(12):
        if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
            explanation = inputs_explanation[input_idx]
            value = inputs_value[input_idx]
            # Color code radar values: red for close obstacles, green for clear
            color = WHITE
            if value > 0.8:  # Very close obstacle
                color = (255, 100, 100)  # Light red
            elif value > 0.5:  # Medium distance obstacle
                color = (255, 255, 100)  # Yellow
            elif value > 0.2:  # Far obstacle
                color = (100, 255, 100)  # Light green
            else:  # Clear path
                color = (100, 255, 100)  # Light green

            draw_text(f"  {explanation}: {value:.3f}", font_small, color)
        input_idx += 1
    y_offset += 5

    # Top 1 Closest Asteroid (3 inputs) - NEW SECTION
    draw_text("CLOSEST ASTEROID:", font_medium, (255, 100, 100), True)
    closest_asteroids = get_closest_asteroid_info(ship, asteroids, top_n=1)
    for i, asteroid in enumerate(closest_asteroids):
        draw_text(
            f"  Asteroid {i + 1}:",
            font_small,
            (255, 150, 150),
        )

    for i in range(6):
        if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
            explanation = inputs_explanation[input_idx]
            value = inputs_value[input_idx]

            # Color code based on input type and value
            color = WHITE
            if "Distance" in explanation:
                # Color code distance: red for close, green for far
                if value > 0.8:  # Very close asteroid
                    color = (255, 100, 100)  # Light red
                elif value > 0.5:  # Medium distance
                    color = (255, 255, 100)  # Yellow
                else:  # Far asteroid
                    color = (100, 255, 100)  # Light green
            else:  # Angle components (sin/cos)
                color = (255, 200, 200)  # Light red for angles

            draw_text(f"  {explanation}: {value:.3f}", font_small, color)
        input_idx += 1
    y_offset += 5

    # Top 3 Closest Minerals (9 inputs)
    draw_text("TOP 3 CLOSEST MINERALS:", font_medium, (100, 255, 100), True)
    closest_minerals = get_closest_mineral_info(ship, minerals, top_n=3)
    draw_text(
        f"  Found {len(closest_minerals)} minerals",
        font_small,
        (150, 255, 150),
    )

    for mineral_idx in range(3):  # Always show 3 mineral slots
        draw_text(f"  Mineral {mineral_idx + 1}:", font_small, (150, 255, 150))

        # Each mineral has 3 inputs: distance, sin(angle), cos(angle)
        for i in range(3):
            if input_idx < len(inputs_explanation) and input_idx < len(inputs_value):
                explanation = inputs_explanation[input_idx]
                value = inputs_value[input_idx]

                # Color code based on input type and value
                color = WHITE
                if "Distance" in explanation:
                    # Color code distance: green for close, red for far
                    if value > 0.7:  # Close mineral
                        color = (100, 255, 100)  # Light green
                    elif value > 0.3:  # Medium distance
                        color = (255, 255, 100)  # Yellow
                    else:  # Far mineral or padded
                        color = (255, 150, 150)  # Light red
                else:  # Angle components (sin/cos)
                    color = (200, 200, 255)  # Light blue for angles

                draw_text(f"    {explanation}: {value:.3f}", font_small, color)
            input_idx += 1
        y_offset += 2  # Small gap between minerals

    draw_text(
        f"TOTAL INPUTS: {len(inputs_value)} (Expected: 28)",
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
        2.0,
        2.0,
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
        dx, dy = 0, 0
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
        if keys[pygame.K_DOWN]:
            # Reverse direction
            dx = -ship.speed * math.cos(ship.angle)
            dy = -ship.speed * math.sin(ship.angle)
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
        if keys[pygame.K_SPACE]:
            inputs_explanation, inputs_value = get_neat_inputs(
                ship, minerals, asteroids
            )
            # Print all inputs for debugging
            for i, (explanation, value) in enumerate(
                zip(inputs_explanation, inputs_value)
            ):
                print(f"Input {i}: {explanation} = {value:.3f}")
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
                # running = False
                print("Collision with asteroid! Game Over!")

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
            pygame.draw.circle(
                debug_screen, WHITE, (int(ship.x), int(ship.y)), 200, 1
            )  # Max radar range

            # RADAR SCAN VISUALIZATION            # RADAR SCAN VISUALIZATION
            N_DIRECTIONS = 12  # Number of radar directions
            MAX_RANGE = 200.0  # Maximum radar range
            radar_scan_results: list[RadarScanResult] = radar_scan(
                ship, asteroids, n_directions=N_DIRECTIONS, max_range=MAX_RANGE
            )

            for i, result in enumerate(radar_scan_results):
                # Calculate absolute angle for visualization
                absolute_angle = ship.angle + result.angle

                # Calculate end point based on surface distance + ship radius
                actual_beam_distance = result.distance + ship.radius

                if result.distance < MAX_RANGE:
                    # Found an asteroid - draw red line to collision point
                    end_x = ship.x + actual_beam_distance * math.cos(absolute_angle)
                    end_y = ship.y + actual_beam_distance * math.sin(absolute_angle)
                    pygame.draw.line(
                        debug_screen,
                        RED,
                        (int(ship.x), int(ship.y)),
                        (int(end_x), int(end_y)),
                        2,
                    )
                    # Draw collision point
                    pygame.draw.circle(
                        debug_screen, (255, 255, 0), (int(end_x), int(end_y)), 3
                    )
                else:
                    # No obstacle - draw green line to max range
                    end_x = ship.x + MAX_RANGE * math.cos(absolute_angle)
                    end_y = ship.y + MAX_RANGE * math.sin(absolute_angle)
                    pygame.draw.line(
                        debug_screen,
                        GREEN,
                        (int(ship.x), int(ship.y)),
                        (int(end_x), int(end_y)),
                        1,
                    )

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

            # # Draw lines to minerals with relative angles
            # mineral_info = get_closest_mineral_info(ship, minerals, top_n=3)
            # for i, info in enumerate(mineral_info):
            #     # Use the actual mineral position for direct line (wrapped if needed)
            #     mineral = info.mineral

            #     # Calculate wrapped distance for visualization
            #     dx = mineral.x - ship.x
            #     dy = mineral.y - ship.y

            #     # Handle screen wrapping for visualization
            #     if dx > WIDTH / 2:
            #         dx -= WIDTH
            #     elif dx < -WIDTH / 2:
            #         dx += WIDTH

            #     if dy > HEIGHT / 2:
            #         dy -= HEIGHT
            #     elif dy < -HEIGHT / 2:
            #         dy += HEIGHT

            #     # Calculate the wrapped angle for visualization
            #     wrapped_angle = math.atan2(dy, dx)

            #     # Draw line showing the relative angle (what the AI sees)
            #     rel_end_x = ship.x + 80 * math.cos(ship.angle + info.relative_angle)
            #     rel_end_y = ship.y + 80 * math.sin(ship.angle + info.relative_angle)
            #     pygame.draw.line(
            #         debug_screen,
            #         RED,
            #         (int(ship.x), int(ship.y)),
            #         (int(rel_end_x), int(rel_end_y)),
            #         2,
            #     )

            #     # Draw line using wrapped angle (should match the relative angle line)
            #     wrapped_end_x = ship.x + 80 * math.cos(wrapped_angle)
            #     wrapped_end_y = ship.y + 80 * math.sin(wrapped_angle)
            #     pygame.draw.line(
            #         debug_screen,
            #         GREEN,
            #         (int(ship.x), int(ship.y)),
            #         (int(wrapped_end_x), int(wrapped_end_y)),
            #         1,
            #     )

            # # Draw lines to asteroids with relative angles
            # asteroid_info = get_closest_asteroid_info(ship, asteroids, top_n=5)
            # for i, info in enumerate(asteroid_info):
            #     # info.angle is already the relative angle from get_closest_asteroid_info
            #     # So we need to add it to ship.angle to get the absolute direction

            #     # Draw line showing the relative angle (what the AI sees)
            #     rel_end_x = ship.x + 80 * math.cos(ship.angle + info.relative_angle)
            #     rel_end_y = ship.y + 80 * math.sin(ship.angle + info.relative_angle)
            #     pygame.draw.line(
            #         debug_screen,
            #         RED,
            #         (int(ship.x), int(ship.y)),
            #         (int(rel_end_x), int(rel_end_y)),
            #         2,
            #     )

            #     # Draw line directly to the asteroid (absolute direction for reference)
            #     asteroid = info.asteroid
            #     pygame.draw.line(
            #         debug_screen,
            #         WHITE,
            #         (int(ship.x), int(ship.y)),
            #         (int(asteroid.x), int(asteroid.y)),
            #         1,
            #     )

            #     # Add text labels
            #     font_small = pygame.font.SysFont(None, 20)
            #     rel_text = font_small.render(
            #         f"Rel: {math.degrees(info.relative_angle):.1f}°",
            #         True,
            #         (255, 255, 255),
            #     )
            #     debug_screen.blit(rel_text, (int(rel_end_x + 5), int(rel_end_y)))

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
