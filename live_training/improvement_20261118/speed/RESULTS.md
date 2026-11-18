# Training throughput

## Complete training for the selected search player

Starting from the frozen game-playing teacher, the complete training recipe ran **1.10× faster**, from **56.19 to 61.78 games/s**. It produced the empirical opponent hand frequencies, fitted continuation policy and conditional opponent-card model. Three paired trials on one physical CPU core completed in 189.64 seconds before optimization and 172.48 seconds after optimization, a **9.05% reduction**. Each seed improved, with throughput ratios of 1.104×, 1.101× and 1.093×. [Per-seed results, elapsed times and artifact hashes](selected_training.json)

| Full selected recipe, summed across three trials | Original | Optimized |
| --- | ---: | ---: |
| Collect single- and four-view exploratory games | 51.80 s | 45.85 s |
| Fit hand frequencies, including all-Random calibration games | 4.27 s | 4.30 s |
| Fit the continuation policy | 3.80 s | 3.81 s |
| Collect labeled hands with single-, four- and 24-view teachers | 125.62 s | 114.38 s |
| Fit and validate the conditional card model | 4.15 s | 4.14 s |
| **Total** | **189.64 s** | **172.48 s** |

Each trial ran 3,552 games, exactly one twenty-fourth of the selected player's 85,248-game training workload. It included 1,728 exploratory games, 192 all-Random calibration games and 1,632 games for conditional hand labels. Those labels retained the original 60 shards with one suit view, 24 with four views and 18 with all 24 views, with 16 games per shard instead of 384. The normal two-, three- and four-player rotation, 18% position sampling, whole-shard holdout and 14 fitting iterations were used throughout. The smaller collection batches affect throughput, so the absolute times apply to this scaled run.

Each implementation processed 10,656 games, 131,046 exploratory decisions, 191,639 teacher-action comparisons, 28,321 training opponent hands and 5,955 held-out opponent hands containing 29,378 cards. Every underlying array in both collections, all three fitted files and the complete conditional validation report matched byte for byte per seed. An independent replay with the production fitting commands reproduced the same files. [Production-command verification](selected_fit_verification.json)

The before and after runs alternated order and used identical prewarmed Numba kernels. Times include process startup, collection, normal validation and artifact writes. Average utilization of all other logical CPUs ranged from 0.19% to 0.35% across the six runs. A preliminary pair straddled the end of seven other CPU jobs. It independently matched all outputs and was excluded from the timed repetitions above. [Preliminary pair](selected_training_transition.json)

## Public temporal PPO

The public temporal PPO trainer completed its steady 768-game epochs **1.34× faster**, from 275.03 to 368.68 games/s. This includes both rollout and learner updates. Each of the five paired training seeds produced identical actor parameters after every epoch and identical final Adam state.

| Public PPO measurement | Original | Optimized |
| --- | ---: | ---: |
| Seconds per steady 768-game epoch | 2.792 | 2.083 |
| Games per second | 275.03 | 368.68 |
| Controlled decisions per second | 13,330 | 17,869 |
| Five complete processes including startup, evaluation, writes and shutdown | 98.83 s | 78.53 s |

The complete processes ran **1.26× faster**. Their five steady-state speedup ratios were 1.38×, 1.36×, 1.21×, 1.32× and 1.45×. Measurements used a Ryzen 7 5800X3D with four physical cores assigned to the trainer and an RTX 3090. The measurements reflect wall time on a shared host with this CPU-rollout, CUDA-learner configuration.

## Work and numerical equivalence

Each implementation ran 25 epochs, 19,200 games, 933,829 controlled decisions and 1,200 Adam updates with the same 192×2 MinGRU teacher, 2/3/4-player schedule and two PPO passes. Opponents cycled between current self-play and canonical RandomAgent. Seed pairs alternated the execution order. The PPO timing repetitions used distinct optimizer seeds and overlapping deal-seed windows. The steady result sums wall times for the three middle epochs in each run, covering 11,520 games and 558,341 controlled decisions. The first epoch initializes workers and kernels. The last epoch runs the same three small evaluation matchups on each side. Both are included in the complete-process result.

The benchmark checked deal/turn/decision/work counts, every epoch's actor state hash, final actor tensors and all final optimizer tensors. Torch deterministic algorithms, float32, highest matmul precision and the same cuBLAS workspace configuration were used on both sides. Raw times and parity checks for every pair are in [training.json](training.json).

The collector fills the five public-event feature columns directly in zero-initialized rows. Recurrent state splitting and selected log-probability gathering now work on batches. One-sample categorical draws use each seat's original CPU generator, with the final probability/noise argmax computed as a batch. FP32 MinGRU worker graphs are traced once and read refreshed shared weights in place. A separate comparison of full packed trajectories at two, three and four players matched the tokens, masks, actions, probabilities, values and outcomes byte for byte. [Packed trajectory digests](packed_parity.json)

## PPO with the default privileged critic

With the default, compiled actor update and the default causal MLP critic, complete steady training epochs ran **1.43× faster**. Each seed started from the identical actor and critic checkpoints. Both arms used separate, initially byte-identical compiler caches, one dedicated CPU core and the RTX 3090. All six runs recorded only graph-cache hits, covering 26 AOTAutograd and 39 Inductor FX hits per implementation. Every scheduled pair qualified for the warm comparison on its first attempt.

| Compiled privileged PPO measurement | Original | Optimized |
| --- | ---: | ---: |
| Seconds per steady 256-game epoch | 10.155 | 7.099 |
| Games per second | 25.21 | 36.06 |
| Controlled decisions per second | 1,239 | 1,772 |
| Three complete processes with warmed compiler caches | 173.12 s | 127.45 s |

Across three disjoint seed pairs, each implementation ran 3,840 games, 187,994 controlled decisions, 63 actor steps and 243 critic steps. All 15 per-epoch actor parameter hashes and all critic loss summaries matched. The final actor, critic and both optimizer states matched per seed. Individual steady speedups were 1.41×, 1.42× and 1.46×. The complete warm-cache processes ran **1.36× faster** including startup, first-epoch tracing, both updates, evaluation and writes. [Compiled training timings, graph-cache counters and parity](privileged_warm_compiled.json)

A separate five-epoch pair on a shared CPU core started each arm with an empty compiler cache. The full processes took 161.87 and 161.11 seconds. Generating the first two epochs' actor-update graphs took most of the process time. All numerical checks also passed in that run. [Cold-compilation run](privileged_compiled.json)

### Eager actor updates

Training with the same critic and eager actor updates ran **1.51× faster** on complete steady epochs. The 256-game workload covered two, three and four players, split evenly between current self-play and RandomAgent. Each of three disjoint seed pairs ran both implementations in alternating order on one CPU core and the RTX 3090. The learner used the default BF16 actor precision and one actor and one critic pass. All PPO comparisons used deterministic Torch and cuBLAS settings for strict parameter comparisons.

| Privileged PPO measurement | Original | Optimized |
| --- | ---: | ---: |
| Seconds per steady 256-game epoch | 10.485 | 6.946 |
| Games per second | 24.42 | 36.86 |
| Controlled decisions per second | 1,223 | 1,846 |
| Three complete processes, including startup, evaluation and checkpoint writes | 164.41 s | 112.85 s |

Each implementation processed 3,840 games, 189,000 controlled decisions, 227,941 game turns, 63 actor and 243 critic optimizer steps. Individual steady speedup ratios were 1.44×, 1.55× and 1.54×. The first and final epochs are included in the **1.46×** complete-process timing. All per-epoch actor parameters, critic loss metrics, final actor and critic parameters, and both complete optimizer states matched exactly. [Timing and numerical parity](privileged_training.json)

Snapshot capture now reuses each seat’s encoded recurrent memory until it next acts. It also shares fixed relative-seat orderings, avoids redundant feature conversions, and traces the in-process MinGRU rollout actor. A separate audit matched every sampled action, log-probability, value, encounter outcome and all 22 snapshot fields byte for byte across 144 games and 6,754 controlled decisions. [Snapshot parity by player count](privileged_snapshot_parity.json)

## Training the search player's hand frequencies and continuation policy

The training stages shared by the online-search candidates ran **1.19× faster**, from 62.82 to 74.45 games/s end to end. Three single-core CPU trials used matched disjoint deal seeds, alternating the original and optimized collectors. Each trial collected six 192-game shards with the raw teacher at temperature 2 and 4% exploration, and three 192-game shards with four suit views at temperature 1.65 and 2.5% exploration. The hidden-hand frequency fit added 64 canonical RandomAgent games per table size. This preserves the full campaign's 2:1 shard mix, 2/3/4-player rotation, exploration parameters and proportion of all-Random calibration games at one twenty-fourth of its game count per trial.

| Complete search-training stages, summed across three seeds | Original | Optimized |
| --- | ---: | ---: |
| Raw-teacher exploratory game collection | 39.60 s | 31.91 s |
| Four-view exploratory game collection | 35.86 s | 32.30 s |
| Fit hand frequencies, including the RandomAgent calibration games | 8.76 s | 6.40 s |
| Fit linear continuation weights | 7.48 s | 6.75 s |
| Total including subprocess startup and file writes | 91.69 s | 77.37 s |

Each implementation ran 5,184 exploratory games and 576 all-Random games. The fits consumed 130,712 focal decisions, 120,938 observed focal hands, 41,800 random hands and 191,496 legal teacher-action comparisons. All exploratory trajectory arrays, hand-frequency JSON files and continuation-weight NumPy files matched byte for byte. Both fits used the same source, linear systems and a shared, prewarmed Numba cache. Their individual timing differences include host variation. Per-seed total speedups were 1.20×, 1.17× and 1.19×. [Matched timings, work counts and output hashes](search_training.json)

## Outcome residual and paired-intervention training

The complete single-core CPU recipe ran **1.27× faster**, from 93.92 to 118.93 games/s. It collected exploratory games, fitted the public value model and action head, collected paired action interventions, then fitted the final residual head. Three disjoint training-seed pairs ran in alternating order. Each implementation processed 6,912 games, 87,000 exploratory decisions and 983 usable interventions. The training collector traced the raw CPU teacher once per run and then applied the requested suit orbit.

| Residual training measurement, totaled over three runs | Original | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Complete processes, including startup and writes | 73.59 s | 58.12 s | **1.27×** |
| Exploratory game collection | 32.34 s | 25.11 s | 1.29× |
| Paired-intervention game collection | 30.62 s | 21.86 s | 1.40× |

Every output array in every deal shard, both sets of learning curves, and both selected heads per seed matched exactly. Complete-process wall-time ratios were 1.45×, 1.18× and 1.17×. These runs used six 192-game exploratory shards and six 96-pair intervention shards per seed, spanning all three table sizes. The fits used 7 value epochs, 8 exploratory-head epochs and 18 paired-head epochs. The timed collector used the one-view teacher. Separate checks covered both actual collectors with 1, 4 and 24 suit views. [Paired timing and hashes](traced_outcome_training.json) · [Suit-view data parity](collector_parity.json)

An earlier measurement of the shared collector changes alone kept the teacher eager. Across 10,368 exploratory games per implementation it recorded a 1.03× cold collect-and-fit speedup and one slower repetition. The combined training result above includes the trace added after that measurement. [Exploratory-only timing](outcome_training.json)

## Regression checks

The native runtime passed 89 tests across the temporal batch collector, worker pool, RandomAgent adapter, public and privileged training, native policy, reward handling, privacy contract and fast trainer. The new tests compare each seat’s RNG position and sampled action after mixed-size batches and verify the traced graphs read fresh shared weights with canonical checkpoint keys. The additional privileged-critic regression run passed all 66 tests across six modules, including the CUDA collation and BF16 optimizer tests. The new cache test covers noncontiguous recurrent tensors, missing seats, mismatched widths, per-seat replacement and the immutability of prior snapshots.

## Reproduction

The source archive and reverse patch reconstruct the starting working-tree versions in a temporary directory. A second patch restores the measured privileged PPO changes. Both source sets must match the recorded SHA-256 hashes before either implementation runs. The residual benchmark also includes a frozen copy of both training recipes. Run from the repository root with fresh scratch and output paths. The CPU search training used the pinned Numba runtime. [Hardware, software versions and base repository revision](environment.json)

```sh
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark.py \
  --scratch /tmp/pesten-public-ppo-comparison --output /tmp/pesten-public-ppo.json
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark_privileged.py \
  --scratch /tmp/pesten-privileged-ppo-comparison --output /tmp/pesten-privileged-ppo.json
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark_privileged.py \
  --compiled --cpus 5 --seeds 563060100,563080100 \
  --scratch /tmp/pesten-compiled-prime --output /tmp/pesten-compiled-prime.json
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark_privileged.py \
  --compiled --require-hot --warm-from /tmp/pesten-compiled-prime --cpus 5 \
  --seeds 563060100,563080100,563100100 \
  --scratch /tmp/pesten-compiled-ppo-comparison --output /tmp/pesten-compiled-ppo.json
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark_selected_training.py \
  --scratch /tmp/pesten-selected-training --output /tmp/pesten-selected-timing.json --cpu 0
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/verify_selected_fits.py \
  --scratch /tmp/pesten-selected-training --timing /tmp/pesten-selected-timing.json \
  --output /tmp/pesten-selected-fit-verification.json
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark_search_training.py \
  --scratch /tmp/pesten-search-training-comparison --output /tmp/pesten-search-training.json
puffer4-native/.venv/bin/python live_training/improvement_20261118/speed/benchmark_outcome.py \
  --scratch /tmp/pesten-residual-comparison --output /tmp/pesten-residual.json \
  --cpu 2 --chunks 6 --games 192 --paired --seeds 543000100,543020100,543040100
```
