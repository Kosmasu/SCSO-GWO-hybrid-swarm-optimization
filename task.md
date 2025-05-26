# Space Miner
A game where the player controls a spaceship that must collect minerals while avoiding asteroids. The ship has limited fuel and must balance mining with survival. We will use NEAT (NeuroEvolution of Augmenting Topologies) to train an AI to play the game automatically.
How It Works
1.	Game Mechanics
    * The spaceship must collect minerals while avoiding asteroids.
    * Mining refuels the ship, but fuel depletes over time.
2.	Neuroevolution AI
    * Uses NEAT to evolve neural networks.
    * Inputs: Distance/angle to minerals, distance to asteroids, fuel level. This is the preliminary suggestion, you should change this.
    * Outputs: Steering, thrust, and mining actions.
    * Fitness function rewards both survival time and minerals collected.
3.	Training Process
    * Runs for 50 or more generations.
    * The best-performing networks will learn to navigate, mine efficiently, and avoid obstacles.

# Notes:
## Angle
Right (East) → 0°
Bottom (South) → 90°
Left (West) → 180°
Top (North) → 270°

## Ship Angle Movement
### Sigmoid Output Range
The sigmoid activation function outputs values in the range **[0, 1]**, not [-1, 1].

### Current Code Analysis
```python
ship.angle += (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
```

Here's what happens:
1. `output[0]` comes from sigmoid: **[0, 1]**
2. `output[0] * 2`: **[0, 2]**
3. `(output[0] * 2 - 1)`: **[-1, 1]**
4. `(output[0] * 2 - 1) * 0.1`: **[-0.1, 0.1]**

### How It Works in Practice

| Sigmoid Output | After Transformation | Angle Change | Turn Direction |
|----------------|---------------------|--------------|----------------|
| 0.0 | -0.1 | -0.1 rad | Left (counter-clockwise) |
| 0.25 | -0.05 | -0.05 rad | Slight left |
| 0.5 | 0.0 | 0.0 rad | No turn |
| 0.75 | +0.05 | +0.05 rad | Slight right |
| 1.0 | +0.1 | +0.1 rad | Right (clockwise) |

### The Transformation Logic
The code correctly maps sigmoid's [0,1] output to a turning range:
- **0.0-0.5**: Turn left (negative angle change)
- **0.5**: Go straight (no turn)
- **0.5-1.0**: Turn right (positive angle change)

This gives the AI smooth, continuous control over turning, where the network can learn to output values near 0.5 for straight movement, or values closer to 0 or 1 for sharper turns in either direction.

The 0.1 multiplier limits the maximum turn rate to ±0.1 radians per frame, preventing overly erratic movement.

# Task
Use NEAT to train the Game agent. The basic code is provided to you including the NEAT training code. The `miner_neat2.py` file is NEAT agent learning to play the game. However, you need to modify `miner_neat2.py` as it is not competently train. You need to change the Inputs to the neural network so that it can learn to play the game well.
