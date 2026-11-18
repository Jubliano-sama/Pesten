# Conditional opponent-hand model

The model estimates which unseen physical cards an opponent holds. It learns a correction to the empirical card-category frequencies from complete training games. At play time it reads the focal hand, public hand sizes, direction-relative opponent position, and top suit and rank. It scores each of the strong and random continuation hypotheses independently. The search sampler draws without replacement from the remaining deck and continues to use the public discard, draw and suit-declaration history.

The fit covered 32,256 training games. Another 6,912 games supplied 48,079 held-out opponent hands and 238,026 cards. Against the size- and style-conditioned category-frequency model, held-out negative log likelihood fell from 3.7485 to 3.7203 nats per card. Both estimators exclude the observer’s own cards and the current top card. The six table-size and opponent breakdowns, training seeds and data hashes are in [fit.json](fit/fit.json).

A second diagnostic played 2,304 fresh games using the full public-history model inherited by both policies. Under this stronger conditioning, mean negative log likelihood fell from 3.3764 to 3.3740 nats on the 247,202 cards present in the inferred sampling pool. The pile estimate excluded 533 observed cards in both policies. Conditioning on public history accounts for most of the standalone predictive gain. [Tracked-history diagnostic](history_validation.json)

The play profile combines this model with 128 sampled worlds, a 72-turn cap, a learned continuation policy and the frozen Orbit24 teacher. It shares all other settings with the control in `lookahead/configs/pile_128_rollout.json`. [Playable profile](profile.json)

## Reproduce the fitted model

Run from the repository root with `puffer4-native/.venv/bin/python`. The native environment’s `search` extra supplies the compiled CPU kernels. The following commands recreate the three source-game batches from the registered seeds. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for the single-core configuration.

```sh
PY=puffer4-native/.venv/bin/python
CAMPAIGN=live_training/improvement_20261118
"$PY" train_public_leaf.py collect --data /tmp/pesten-card-belief-shards --orbit 1 --seed 1111000000 --start 0 --chunks 60 --games 384 --keep .18
"$PY" train_public_leaf.py collect --data /tmp/pesten-card-belief-shards --orbit 4 --seed 1111000000 --start 60 --chunks 24 --games 384 --keep .18
"$PY" train_public_leaf.py collect --data /tmp/pesten-card-belief-shards --orbit 24 --seed 1114000000 --start 84 --chunks 18 --games 384 --keep .18
"$PY" train_card_belief.py --data /tmp/pesten-card-belief-shards --prior "$CAMPAIGN/lookahead/hand_frequencies.json" --output /tmp/pesten-card-belief-fit --shards 102 --iterations 14
```

## Fit with public histories

The second fit learns the same compact correction on the actual public-history sampler. It uses the remaining-card multiplicities, the inferred discard pile, and the recorded public draw and declaration features in the training likelihood. Across 7,488 training and 2,880 separate validation games, the validation loss improved from 3.3428 to 3.3297 nats per supported card. Each of the six cells improved. The tournament used the first fit. [History-conditioned fit](fit_history/fit.json)

```sh
puffer4-native/.venv/bin/python live_training/improvement_20261118/card_belief/fit_history.py \
  --prior live_training/improvement_20261118/lookahead/hand_frequencies.json \
  --output /tmp/pesten-tracked-history-card-fit --chunks 54
```

On a replay of 144 real positions with multiple card or suit choices, the tested conditional planner took 5.62 ms per choice. The otherwise identical control took 5.76 ms. Four alternating warmed timing pairs used one CPU thread while another job used its physical core’s sibling. Both configurations include the learned continuation policy. These timings cover the planner component. [Timing results](planner_cost.json)

## Development tournament

Each focal policy played 2,880 games across 160 independent, seat-rotated deals per cell. Each cell contributed equally to the aggregate. The same six seed banks were used for all three policies.

| Other seats | Players | Incumbent | Search control | With conditional hands |
| --- | ---: | ---: | ---: | ---: |
| Frozen RL | 2 | 50.00% | 52.50% | 55.31% |
| Frozen RL | 3 | 33.12% | 35.62% | 36.88% |
| Frozen RL | 4 | 24.84% | 25.78% | 25.47% |
| AgentRandom | 2 | 76.25% | 81.25% | 82.81% |
| AgentRandom | 3 | 59.79% | 58.75% | 57.29% |
| AgentRandom | 4 | 40.31% | 40.16% | 42.34% |
| **Equal mean** | | **47.39%** | **49.01%** | **50.02%** |

The paired improvement over the incumbent was 2.63 percentage points (95% interval +0.94 to +4.32). Adding the conditional hand model to the same search improved the observed mean by 1.01 points (−0.86 to +2.88). [Development games](tournament/rows.jsonl) · [Recurrent incumbent comparison](tournament/summary.json) · [Paired control comparison](paired_control.json)

```sh
puffer4-native/.venv/bin/python live_training/improvement_20261118/screen.py \
  --profile fixed lookahead live_training/improvement_20261118/lookahead/configs/pile_128_rollout.json \
  --profile conditional lookahead live_training/improvement_20261118/card_belief/profile.json \
  --seed 1643100000 --blocks 160 --chunk 32 --workers 1 --output /tmp/pesten-belief-tournament
```
