# Pesten-AI: An AI for the Dutch Card Game "Pesten"

This repository contains a Python implementation of the Dutch card game "Pesten" (similar to Crazy Eights or Mau-Mau) and a collection of AI agents designed to play it. The project ranges from a simple game engine and rule-based agents to a sophisticated Reinforcement Learning pipeline using Proximal Policy Optimization (PPO) and knowledge distillation.

## ✨ Features

*   **Complete Game Engine**: A fully functional, customizable game engine (`game.py`) that handles all the rules of Pesten, including special cards, penalties, and turn management.
*   **Diverse AI Agents**:
    *   **Random Agent**: A baseline agent that plays a random valid card.
    *   **Smart Agent**: A rule-based agent using a set of tunable heuristics to make decisions.
    *   **Bayesian Tree-Search Agent**: A deterministic information-set agent
        that updates beliefs about hidden hands from public play history and
        searches plausible replies from every player.
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
| `hardcodedAgent.py`     | Bayesian hidden-hand sampling plus bounded multi-player tree search.                                    |
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

The `--agents` flag takes a comma-separated list of agent types: `human`,
`random`, `smart`, `hardcoded` (or `tree`), `supervised`, and `best`. Names
are validated exactly; an unknown name is an error and is never replaced by a
random player.

To play against the selected production player:

```bash
python interactive_game.py --agents human,best
```

`best` loads `ppo_actor_100run3.pth` through `best_player.py`, verifies its
SHA-256 is
`aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186`,
freezes it in greedy evaluation mode, and applies the production projection:
all 24 ordinary suit relabelings averaged in logit space, with no player-label
orbit averaging. The draw-retry adapter permits exactly the newly drawn card
or pass on the immediate retry. If the checkpoint is absent or its digest does
not match, startup fails explicitly instead of selecting another model.
On CPU, the frozen base network is traced once at load time; an eager opt-out
is available through `BestPlayerConfig(trace_cpu_actor=False)`. The tracked
trace audit requires bit-identical projected logits and actions.

To verify the tracked champion, one-way claim ledger, raw sealed records,
summary, and report without running any games:

```bash
python verify_autoresearch_release.py
```

Add `--verify-bootstrap-cis` to replay the recorded confidence intervals with
bounded memory. The verifier imports no game engine or benchmark evaluator.

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
*   `--torch-threads`: PyTorch CPU thread count. The profiled default is 8;
    override it when the host has a materially different CPU topology.

#### Causal omniscient asymmetric critic

The temporal PPO pipeline defaults to a separate environment-side critic that
sees all current-turn cards and engine state while leaving the actor observation
unchanged. The gamma-0.995 1.07M-parameter MLP was trained across 2-8 players
and is loaded automatically by `train_puffer_fast.py`:

```bash
python train_puffer_fast.py \
  --resume models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt \
  --output-dir live_training/puffer_privileged
```

Use `--privileged-critic CHECKPOINT` to override it, or
`--no-privileged-critic` to fall back to the actor's public value head. The
default discount is `gamma=0.995`, so a terminal component retains about
90.461% of its value across 20 controlled-seat discount steps (0.5%
depreciation per step). The separate `-0.01` per-decision reward remains part
of the return. The gamma sweep did not statistically resolve 0.995 as stronger
than 0.99; 0.995 is the explicit operational default selected after that
evaluation.

The external value is used for GAE and updated only from retained pre-action
snapshots; private fields never enter actor tokens. See
[the 2026-08-14 investigation](PRIVILEGED_CRITIC_20260814.md) for the exact
causal contract, feature findings, held-out results, and limitations.

#### Iterative-reference two-player self-play

`nashpg_ppo.py` is a development-only, shared-symmetric specialization of
NashPG. It removes legacy pass shaping, audits terminal two-player payoffs,
uses `KL(current || reference)`, freezes the reference for each inner block,
and refreshes it only between outer rounds.

```bash
python nashpg_ppo.py \
  --output-dir experiments/nashpg-alpha02 \
  --alphas 0.2 \
  --outer-rounds 8 \
  --inner-iterations 5 \
  --games 256 \
  --update-epochs 3 \
  --learning-rate 3e-5
```

Use `benchmark_nashpg_crossplay.py` to evaluate every finalized outer
checkpoint on fresh, explicit development seeds. It compares all neural pairs
in both seats and uses common deals against random, Smart, and Smarter:

```bash
python benchmark_nashpg_crossplay.py \
  --nashpg-manifest experiments/nashpg-alpha02/nashpg_pilot_metrics.json \
  --stage screen \
  --pairs 100 \
  --base-seed 89000000 \
  --output benchmarks/nashpg-alpha02-screen.json
```

#### Exact suit-projected continuation

`projected_ppo.py` trains through the same exact 24-suit mean-logit policy used
for rollout, update, and production inference. It audits the initial behavior
ratio, terminal zero-sum payoff, replayed deal seeds, and a full-batch target
KL; an overshooting epoch restores the actor, critic, and both optimizers
atomically. `projected_ppo_iterative.py` repeats that update on fresh rollouts:

```bash
python projected_ppo_iterative.py \
  --output-dir benchmarks/projected-ppo-iterative \
  --iterations 8 \
  --games 256
```

The saved checkpoints contain raw shared base weights and must be wrapped in
the exact suit projection for evaluation or deployment.

#### Top-advantage filtering and EMA

`filtered_ema_ppo.py` is a development-only current-policy self-play lane. The
critic learns from every transition, while the actor can retain only the
highest-advantage fraction; both the online actor and its parameter EMA are
saved for separate evaluation:

```bash
python filtered_ema_ppo.py \
  --output-dir benchmarks/filtered-ema-deep \
  --keep-fractions 0.25 1.0 \
  --iterations 8 \
  --games 256 \
  --critic-warmup-games 256
```

Every training or evaluation range must remain development-only. Use
`benchmark_nashpg_crossplay.py --suit-project all` on fresh explicit seeds
before making any strength claim.

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

#### Bayesian tree search versus a checkpoint

`benchmark_agents.py` alternates seats and reuses each seed once per seat, so
the comparison is not biased toward the starting player. The PPO policy is
evaluated with deterministic masked argmax.

```bash
python benchmark_agents.py \
  --opponent ppo \
  --model ppo_actor_100run3.pth \
  --games 2000 \
  --output benchmarks/tree-vs-ppo.json
```

The tree-search player starts from the uniform hidden-deal prior. Previously
played suits/ranks update a Dirichlet posterior; an announced Jack suit counts
as stronger public evidence. Posterior hand samples are then searched with a
bounded multi-player minimax tree. It never reads another player's actual hand.

#### One persistent Codex player

`luna_benchmark.py` exposes an stdin/stdout game protocol for a single
long-lived model context. Start it once in a PTY, answer every `DECISION` and
`SUIT_DECISION` with one listed token, and keep the same model thread through
all games:

```bash
python luna_benchmark.py \
  --games 20 \
  --output benchmarks/luna-vs-tree.json
```

The protocol alternates seats but uses a fresh, unrevealed deal for every game;
replaying a paired deal would leak the opponent's initial hand to a persistent
model. It writes every public observation and response to an audit JSONL and
reports first-half versus second-half performance so in-context adaptation can
be inspected.

Use the same protocol against the deployed PPO architecture with:

```bash
python luna_benchmark.py \
  --games 20 \
  --opponent ppo \
  --model ppo_actor_100run3.pth \
  --output benchmarks/luna-vs-ppo.json
```

Validate a completed transcript independently:

```bash
python audit_luna_benchmark.py \
  benchmarks/luna-vs-ppo.audit.jsonl \
  --result benchmarks/luna-vs-ppo.json
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
