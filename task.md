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
# Task
Use NEAT to train the Game agent. The basic code is provided to you including the NEAT training code. The `miner_neat2.py` file is NEAT agent learning to play the game. However, you need to modify `miner_neat2.py` as it is not competently train. You need to change the Inputs to the neural network so that it can learn to play the game well.
