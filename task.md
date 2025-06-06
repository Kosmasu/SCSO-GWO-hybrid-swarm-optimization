# Space Miner
## Game Mechanics
1. Player controls a spaceship.
2. Player have two actions:
    a. Steer left and right (angle is in radians)
    b. Move forward or backward
3. Game over if:
    a. Ship collided with asteroid
    b. Ship ran out of fuel
4. Ship will automatically collect mineral when collided

- Ship's fuel will deplete only when the ship is moving.
- Asteroid move on a random direction
- Asteroid have a random radius

## Competition
We need to train a neat bot to control the ship. The environment that the bot will compete on is:
- 10 Asteroids
- Score calculation = (alive_frame / 4) + minerals_mined * 100
## Variables
```py
ASTEROID_MAX_RADIUS: int = 30
ASTEROID_MIN_RADIUS: int = 15
ASTEROID_MAX_SPEED: float = 2.0
ASTEROID_MIN_SPEED: float = -2.0
MINERAL_RADIUS: int = 10
```
