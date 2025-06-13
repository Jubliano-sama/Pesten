You are absolutely right! My apologies for misinterpreting the standard use of distillation. Using it to bootstrap a larger model is a clever approach.

I have revised the `README.md` to accurately reflect that the student model is larger and that distillation is used as a powerful initialization technique.

Here is the corrected version:

---

# Pesten-AI: An AI for the Dutch Card Game "Pesten"

This repository contains a Python implementation of the Dutch card game "Pesten" (similar to Crazy Eights or Mau-Mau) and a collection of AI agents designed to play it. The project ranges from a simple game engine and rule-based agents to a sophisticated Reinforcement Learning pipeline using Proximal Policy Optimization (PPO) and knowledge distillation.

## ✨ Features

*   **Complete Game Engine**: A fully functional, customizable game engine (`game.py`) that handles all the rules of Pesten, including special cards, penalties, and turn management.
*   **Diverse AI Agents**:
    *   **Random Agent**: A baseline agent that plays a random valid card.
    *   **Smart Agent**: A rule-based agent using a set of tunable heuristics to make decisions.
    *   **Supervised/RL Agent (Teacher)**: A powerful neural network-based agent trained using Proximal Policy Optimization (PPO) through self-play and against other agents.
    *   **Distilled Student Agent**: A more powerful agent with a larger architecture, initialized by distilling knowledge from the RL "teacher". This bootstrapping process creates a highly capable final agent.
*   **Reinforcement Learning Pipeline**: A complete PPO training script (`ppo2.py`) to train the neural network agent. It includes features like reward shaping, GAE, and playing against past versions of itself.
*   **Knowledge Distillation for Bootstrapping**: An advanced technique used to initialize a *larger* student model. Instead of model compression, distillation transfers the learned policy of the RL agent to a new architecture, providing a superior starting point.
*   **Comprehensive Evaluation Suite**:
    *   `interactive_game.py`: Play against the AI agents yourself in the terminal.
    *   `simulate_random_game.py`: Run thousands of games in parallel to benchmark agent performance.
    *   `elo.py`: Calculate Elo ratings for all agents to rank their relative strength.
*   **Hyperparameter Optimization**: A script (`optimize_parameters.py`) to automatically tune the heuristics of the `SmartAgent` for maximum win rate.

## 📂 Project Structure

The codebase is organized into several key components:

| File                    | Description                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| **Core Game Logic**     |                                                                                                         |
| `game.py`               | The main game engine that manages state, rules, and player turns.                                       |
| `card.py`               | Defines the `Card` class and its properties.                                                            |
| `deck.py`               | Defines the `Deck` class for managing collections of cards.                                             |
| **AI Agents**           |                                                                                                         |
| `randomAgent.py`        | A simple agent that plays random valid moves.                                                           |
| `smartAgent.py`         | A heuristic-based agent with tunable parameters.                                                        |
| `supervisedAgent.py`    | Defines the neural network architecture (`DenseSkipNet`) and a base agent wrapper.                      |
| **Training & Learning** |                                                                                                         |
| `ppo2.py`               | The main PPO training script for the Reinforcement Learning agent.                                      |
| `distill_rl_agent.py`   | Bootstraps a larger 'student' model by distilling the policy from the RL 'teacher'.                     |
| `distill_critic.py`     | Bootstraps a larger 'student' critic by distilling the value function from the RL 'teacher'.            |
| `optimize_parameters.py`| Optimizes the parameters for the `smartAgent` using gradient ascent.                                    |
| **Evaluation & Tools**  |                                                                                                         |
| `interactive_game.py`   | An interactive command-line interface to play the game.                                                 |
| `simulate_random_game.py`| Runs mass simulations to evaluate agent win rates.                                                      |
| `elo.py`                | Simulates a tournament to calculate Elo ratings for the agents.                                         |

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   NumPy
*   tqdm

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/pesten-ai.git
    cd pesten-ai
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install torch numpy tqdm
    ```

## 🎮 How to Use

This project provides several scripts to play, train, and evaluate the agents.

### 1. Play Against the AI

You can play an interactive game against any combination of agents.

```bash
python interactive_game.py --agents human,smart,random
```

The `--agents` flag takes a comma-separated list of agent types: `human`, `random`, `smart`, `supervised`, `student`.

### 2. Train the RL Agent

The `ppo2.py` script trains the neural network agent using Reinforcement Learning. It will periodically save model checkpoints (`ppo_actor_*.pth`, `ppo_critic_*.pth`).

```bash
# Train the RL agent against the SmartAgent for 100 iterations
python ppo2.py --iterations 100 --episodes 1000 --agent_sets "smart,rl"

# Train against multiple opponent configurations
python ppo2.py --iterations 200 --episodes 1000 --agent_sets "smart,rl;random,rl;self,self,rl"
```

*   `--iterations`: Number of training iterations.
*   `--episodes`: Number of game episodes to generate for data collection in each iteration.
*   `--agent_sets`: Semicolon-separated lists of opponents for training games. `rl` refers to the agent being trained, and `self` refers to another instance of the same agent.

### 3. Distill a Trained Model

After training an RL agent (e.g., `ppo_actor.pth`), you can use distillation to initialize a new, larger student model.

```bash
# Distill the actor (policy) to bootstrap a student model
python distill_rl_agent.py

# Distill the critic (value function) to bootstrap a student model
python distill_critic.py
```

These scripts will load the teacher models (`ppo_actor.pth`, `ppo_critic.pth`) and save the bootstrapped student models (`distilled_rl_agent.pth`, `distilled_critic.pth`).

### 4. Evaluate Agent Performance

#### Mass Simulation

Run a large number of games to get win-rate statistics. The `--parallel` flag is recommended for speed.

```bash
# Simulate 1000 games between the Student and Smart agents
python simulate_random_game.py -n 1000 --agents student,smart --parallel
```

#### Elo Rating

Calculate Elo ratings to rank all agents against each other.

```bash
# Run 5000 total games to calculate Elo ratings
python elo.py --total_games 5000

# Specify which student models to include in the tournament
python elo.py --total_games 5000 --student_files "distilled_rl_agent.pth,ppo_actor_80.pth"
```

### 5. Optimize the Smart Agent

Tune the parameters of the rule-based `smartAgent` to maximize its win rate against the `randomAgent`.

```bash
python optimize_parameters.py --epochs 50 --games 10000
```

## 🤖 The Agents

*   **`RandomAgent`**: The simplest agent. It shuffles its hand and plays the first compatible card it finds.
*   **`SmartAgent`**: A heuristic-based agent that evaluates moves based on a scoring function. Its parameters (`stop_bonus`, `chain_bonus`, etc.) are tuned to improve its strategy.
*   **`RL_SupervisedAgent` (Teacher)**: The agent trained via PPO. It uses a `DenseSkipNet` neural network to map the game state (a 121-dimensional vector) to a policy over all possible actions. It learns complex strategies through extensive self-play.
*   **`StudentAgent`**: A powerful agent featuring a larger neural network architecture. Instead of being trained from scratch, it is **bootstrapped** using knowledge distillation. It learns to mimic the policy of the trained RL "teacher," which provides a highly effective starting point and often results in a more robust and generalized final model.

## 🧠 Pre-trained Models

The repository includes several pre-trained model weights (`.pth` files). These files contain the learned parameters for the neural network agents.

*   `ppo_actor.pth` / `ppo_critic.pth`: Weights for the RL teacher agent and its value function.
*   `distilled_rl_agent.pth` / `distilled_critic.pth`: Weights for the distilled student models.
*   Other files like `ppo_actor_*.pth` are checkpoints saved during the PPO training process.

You can use these pre-trained models for evaluation or as a starting point for further training.

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, agents, or improvements, feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.