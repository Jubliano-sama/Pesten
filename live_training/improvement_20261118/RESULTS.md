# Training and win rates at two, three and four players

The new player won **50.08%** of games averaged equally across the six test configurations. The previous best averaged **47.57%** on the same deals. The gain was **2.51 percentage points** with a paired 95% interval of **+1.96 to +3.05**. The winning recipe trained **1.10× faster** on a matched benchmark covering every stage at 1/24 of its game count. Default compiled PPO trained **1.43× faster** on warm-cache steady epochs and **1.36× faster** across complete warm-cache runs. [Verified match results](final/primary_verified.json) · [Winning recipe timings](speed/selected_training.json) · [PPO timings](speed/privileged_warm_compiled.json)

## Win rate on 57,600 held-out games

The previous-best and new-player columns describe the focal seat. Each of the other seats uses the opponent named in the first column. The interval is the paired 95% confidence interval for the change in percentage points.

| Other seats | Format | Previous best | New player | Change (pp) | 95% interval (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| Previous best RL | 1v1 | 49.97% | 53.88% | +3.91 | +2.41 to +5.40 |
| Previous best RL | 1v1v1 | 33.21% | 36.67% | +3.46 | +2.28 to +4.64 |
| Previous best RL | 1v1v1v1 | 24.64% | 26.83% | +2.19 | +1.26 to +3.11 |
| RandomAgent | 1v1 | 79.69% | 80.50% | +0.81 | −0.77 to +2.39 |
| RandomAgent | 1v1v1 | 55.94% | 59.31% | +3.38 | +1.89 to +4.86 |
| RandomAgent | 1v1v1v1 | 42.00% | 43.31% | +1.31 | +0.10 to +2.53 |
| **Equal-weight average** | | **47.57%** | **50.08%** | **+2.51** | **+1.96 to +3.05** |

Averaged over table sizes, the gain was **3.18 points against the previous best** and **1.83 against RandomAgent**. Five individual intervals lie above zero. The head-to-head RandomAgent interval includes both a small loss and a gain. A separate 20,000-resample bootstrap put the overall gain between +1.95 and +3.04 points. [Raw deal outcomes](final/primary.json) · [Recomputed intervals and win counts](final/primary_verified.json) · [Interval chart](final/win_rate_chart.html)

Giving the two opponents equal weight within each format, the head-to-head win rate rose from 64.83% to 67.19% (+2.36 points), three-player from 44.57% to 47.99% (+3.42) and four-player from 33.32% to 35.07% (+1.75).

## Standalone actor without decision-time search

On a separate 28,800-game test in the original scalar game engine, the trained action head improved the six-configuration average from **47.87% to 49.22%**, a gain of **1.35 points** (paired 95% interval **+0.64 to +2.06**). The observed gains came from two- and three-player games. In four-player games the two opponent types averaged a change of −0.16 points (−1.06 to +0.75). [Raw deal outcomes](final/standalone.json) · [Recomputed intervals and win counts](final/standalone_verified.json) · [Frozen actor](final/standalone.zip)

| Other seats | Format | Previous best | Standalone actor | Change (pp) | 95% interval (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| Previous best RL | 1v1 | 49.94% | 54.06% | +4.13 | +2.02 to +6.23 |
| Previous best RL | 1v1v1 | 33.25% | 35.08% | +1.83 | +0.14 to +3.52 |
| Previous best RL | 1v1v1v1 | 24.75% | 24.50% | −0.25 | −1.42 to +0.92 |
| RandomAgent | 1v1 | 79.81% | 80.94% | +1.13 | −0.83 to +3.08 |
| RandomAgent | 1v1v1 | 56.92% | 58.25% | +1.33 | −0.60 to +3.26 |
| RandomAgent | 1v1v1v1 | 42.56% | 42.50% | −0.06 | −1.44 to +1.32 |

## Mixed-opponent tables

In 7,680 additional games, the focal player faced both opponent types at the same table. Each lineup used 128 fresh deals, every focal seat and every distinct ordering of its opponents. All three intervals include zero. These games are reported separately from the six-configuration average. [Registered lineups](mixed_protocol.json) · [Seat orders and raw outcomes](final/mixed.json) · [Independent interval check](final/mixed_verified.json)

| Other seats | Previous best | New player | Change (pp) | 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| 1 previous best, 1 random | 42.84% | 42.32% | −0.52 | −3.59 to +2.55 |
| 1 previous best, 2 random | 36.33% | 37.17% | +0.85 | −1.57 to +3.26 |
| 2 previous best, 1 random | 28.97% | 30.86% | +1.89 | −0.17 to +3.95 |

## Match setup

The incumbent is the 192 × 2 MinGRU at epoch 4,387, using all 24 suit permutations. Each permutation has its own recurrent history. The policy maps its action logits back into the original suits, averages them, applies the legal-action mask and takes the maximum. The frozen checkpoint has SHA-256 `f8f0aece2d759eadc60b3cbccca30c321c01da13322f7cae511a73c8da3a9f93`. [Public service metadata before the new release](../production_20261118_search/previous_health.json)

Every remaining seat at a table uses the named opponent, either the incumbent or the project's canonical `randomAgent.AgentRandom`. For each deal, the challenger and incumbent each play once in every seat. Games use the canonical `game.Game` first-finisher rules and a 1,000-turn cap. A draw contributes zero outright wins. The six opponent and player-count combinations carry equal weight. [Fixed protocol](protocol.json)

Eleven development policies shared 320 independent deals per configuration. Selection used their average outright win rate across the six configurations. The primary test reserved 1,600 fresh deals per configuration, or 28,800 games per policy. The supplementary standalone-actor test reserved another 800 deals per configuration, or 14,400 games per policy. Both test banks were set aside before selecting the finalists. The primary, standalone and mixed panels together contain **94,080 games**. [Development roster](selection/roster.json) · [Standalone test protocol](selection/standalone_protocol.json)

Intervals use the paired win-rate difference across independent deals. Each deal retains all its seat rotations. The six configurations use disjoint seed ranges. The independent verifier also resamples entire deals within each configuration for a stratified bootstrap check. [Verifier](verify_result.py)

Before the test, the primary evaluator matched the scalar game on wins, losses, draws and turn counts in 54 separate seat-rotated games. For the standalone actor, batch-size rounding made two card logits tie on one decision where the scalar actor distinguished them by about 0.000002. That game took one extra turn in scalar play, with the same winner. The full standalone comparison therefore uses scalar inference in the original game engine. [Primary parity](final/primary_parity.json) · [Standalone parity](final/standalone_parity.json) · [Standalone evaluation decision](final/standalone_method.json)

A direct scalar replay of the first 48 primary test deals in each cell covered 1,728 games and 102,650 turns. It reproduced every paired deal's win, loss and draw counts and all twelve arm-by-cell turn totals. The replayed games are counted once in the benchmark. [Scalar audit](final/primary_scalar_audit.json)

The targeted regression run passed 154 tests. A separate game-engine suite passed 28 tests and 1,152 subtests. [Policy, training and evaluation test log](eval/regression_tests.log) · [Game-engine test log](eval/engine_regressions.log)

## Training throughput

The speed changes batch event encoding and recurrent policy work, preserve each seat's random stream, and reuse unchanged recurrent-memory features in privileged critic snapshots. Timing uses elapsed wall time and equal work before and after each change.

| Workload | Original games/s | Optimized games/s | Speedup |
| --- | ---: | ---: | ---: |
| Complete selected recipe, all data collection and three fits, 1/24 game count | 56.19 | 61.78 | **1.10×** |
| Privileged-critic PPO, compiled updates, warm-cache steady epochs | 25.21 | 36.06 | **1.43×** |
| Public PPO, steady epochs | 275.03 | 368.68 | 1.34× |
| Privileged-critic PPO, eager updates, steady epochs | 24.42 | 36.86 | 1.51× |
| Complete exploratory and paired-intervention training | 93.92 | 118.93 | 1.27× |
| Collect trajectories, fit the hand distribution and fit the simulation policy | 62.82 | 74.45 | 1.19× |
| Collect counterfactual branches and fit the standalone action head | 13.04 | 15.12 | 1.16× |

The public PPO comparison used four physical Ryzen 7 5800X3D cores and an RTX 3090. The privileged comparisons used one physical core and the same GPU. Each collect-and-fit comparison used one CPU core. Each row has its own game mix, updates and starting conditions. The public PPO run reproduced the actor and optimizer, the privileged runs reproduced both networks and optimizers, and the CPU training runs reproduced the collected data and fitted parameters. [Timings, workloads and reproduction commands](speed/RESULTS.md) · [Standalone action-head training measurements](recipe/branch_training_timing.json)

The three compiled PPO runs completed in 173.12 s before and 127.45 s after, a **1.36×** end-to-end speedup including startup, updates, evaluation and checkpoint writes. Each implementation processed 3,840 games and 187,994 controlled decisions. All 15 epoch-end actor hashes matched, along with both final networks and both optimizer states. In a separate five-epoch run starting with empty compiler caches, compilation dominated the total wall time on a shared core (161.87 s and 161.11 s). [Warm compiled runs](speed/privileged_warm_compiled.json) · [Cold compiled check](speed/privileged_compiled.json)

The full recipe uses 85,248 collection and validation games starting from the incumbent checkpoint. The **1.10×** speed comparison scaled every game category to 1/24 and ran three matched seeds. It covered trajectory collection, RandomAgent calibration, continuation-policy fitting, and the separate collection and 14-pass fit for conditional card probabilities. Across 10,656 games per implementation, training time fell from **189.64 to 172.48 seconds**, including startup and output writes for each stage. Both collected datasets, all three fitted artifacts and the conditional validation report matched across implementations for each seed. Individual speedups ranged from 1.093× to 1.104×. A separate invocation of the repository's fitting commands reproduced all nine fitted files and all three validation reports byte for byte. [All-stage timings and fitted hashes](speed/selected_training.json) · [Independent refits](speed/selected_fit_verification.json)

The 1.19× search-training comparison measured the shared trajectory, frequency and continuation stages on 5,760 games per implementation. The 1.16× standalone comparison held its pretrained critic, style classifier and card-frequency inputs fixed, then collected and fitted 24,402 move branches per implementation. [Search-stage timings](speed/search_training.json) · [Action-head timings](recipe/branch_training_timing.json)

## Training recipes

The selected player keeps the incumbent's 24-view policy and samples 128 possible deals from its own hand and public history at eligible decisions. It simulates each competing move for up to 72 turns on matched deals, then adds the estimated win difference to the policy logits. Forced plays and immediate draw retries preserve their legal-action restrictions. A continuation model fitted to 1,550,414 legal-action comparisons supplies simulated play. The conditional hand model learned from 221,900 labeled opponent hands across 32,256 training games. The real opponents' cards are available as training labels only. [Selected profile](card_belief/profile.json) · [Trajectory collection and continuation training](lookahead/RUN.md) · [Opponent-hand training and validation](card_belief/README.md)

The standalone action learner branches the game during training, forces plausible legal alternatives and estimates their outcomes after one, four and twelve turns. A critic trained on eventual winners can use full states in those simulated branches. The player at the table uses its own hand, public play history and a trained action head. [Standalone recipe](recipe/RUN.md) · [Critic training and exact input archive](learned_search/fit2/README.md)

On the 320-deal selection bank the search player gained 3.54 percentage points, and the best standalone action head gained 1.32. A conventional PPO continuation trained for 93 epochs finished 0.26 points below the incumbent. The final estimates use fresh deals. [All eleven development comparisons](selection/comparison.json) · [Selected artifact hashes and protocols](final/selection.json)

## Cost per move

The search player averaged **4.10 ms per decision** on one CPU thread, alongside **2.32 ms** for the incumbent. Its 95th percentile was 8.87 ms versus 4.29 ms. The standalone actor averaged 6.08 ms alongside 2.21 ms for its matched incumbent timing. Both measurements replayed the same 2,951 actual decision histories from 108 scalar games, with all player counts and seat rotations represented. Each timing alternated implementation order across four repetitions and included public-event processing and action selection. [Search timing](selection/learned_cards_rollout/decision_timing.json) · [Standalone timing](selection/actor/belief195_timing.json)

## Play the selected agent

Use the native Python environment with the `search` extra. From the repository root, the following plays a three-player game against two copies of RandomAgent.

```python
import torch
from game import Game
from public_lookahead import load_candidate
from randomAgent import AgentRandom
from temporal_puffer_agent import temporal_agent_factory

torch.set_num_threads(1)
policy = load_candidate("live_training/improvement_20261118/card_belief/profile.json", device="cpu")
challenger = temporal_agent_factory(policy, deterministic=True)
turns, winning_seat = Game([challenger, AgentRandom, AgentRandom]).auto_simulate()
```

The frozen archives include the source modules, fitted artifacts, incumbent and runtime lock. Extract each archive into its own directory. The primary archive's `frozen_candidate.json` gives the commands to replay its 57,600 games. To reproduce the reported standalone and mixed-table results, use `scalar_standalone.py` and `mixed_tables.py` from the scalar replay bundle. Its `README.txt` gives the replay and independent verification commands, with the extracted player directory as `--root`. [Primary archive](final/primary.zip) · [Standalone archive](final/standalone.zip) · [Scalar replay bundle](final/scalar_evaluators.zip) · [Artifact checksums](final/SHA256SUMS)
