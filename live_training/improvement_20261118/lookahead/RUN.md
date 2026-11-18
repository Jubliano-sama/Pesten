# Train and run the public lookahead policy

Run these commands from the repository root. The native environment's `search` extra installs Numba for the CPU rollouts. The pre-trained profile requires the canonical epoch-4387 checkpoint at its repository path.

```bash
uv sync --project puffer4-native --python 3.12 --extra search --frozen
py=puffer4-native/.venv/bin/python
run=live_training/improvement_20261118/lookahead
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Collect training games and fit both components

The 108 training shards contain 41,472 games. Each consecutive set of three shards contains a two-, three-, and four-player table. The focal seat explores with the frozen RL teacher. The other seats use either the greedy teacher or canonical RandomAgent. Shards 0 through 71 use one suit labeling and shards 72 through 107 average over four. Collection seeds start at `510000000 + 384 × shard_index`.

```bash
work=$(mktemp -d)
shards="$work/training"
"$py" train_outcome_residual.py collect --directory "$shards" \
  --seed 510000000 --games 384 --start 0 --chunks 36 --orbit 1 --temperature 2 --epsilon .04
"$py" train_outcome_residual.py collect --directory "$shards" \
  --seed 510000000 --games 384 --start 36 --chunks 36 --orbit 1 --temperature 2 --epsilon .04
"$py" train_outcome_residual.py collect --directory "$shards" \
  --seed 510000000 --games 384 --start 72 --chunks 36 --orbit 4 --temperature 1.65 --epsilon .025
"$py" train_public_lookahead.py "$shards" "$work/hand_frequencies.json" \
  --seed 726000000 --random-games 1536
"$py" "$run/fit_rollout_policy.py" "$shards" "$work/rollout_weights.npy"
```

The hand model uses 963,817 training observations from the exploratory trajectories and 325,532 observations from 4,608 additional canonical RandomAgent games. It estimates card-category frequencies and suit/rank relationships from the hand size, the opponent's last visible card, and its latest suit declaration. The continuation policy fits 1,550,414 legal-action logit differences by weighted ridge regression. It uses three sets of coefficients, one per player count.

The reference artifacts are `hand_frequencies.json` (SHA-256 `9727824e7132f494e1804fafabfe2451c3652aacfe8914aefd83bea140363fd0`) and `rollout_weights.npy` (SHA-256 `3b1dde3c721d23cea5bf26366227d7e53ca7869f944118dd026aff9bdec4b9a4`). Both fits reproduced the exact reference bytes from the retained source shards. The selected JSON profile records its dependencies by path and hash.

## Check imitation on independent games

```bash
"$py" train_outcome_residual.py collect --directory "$work/validation" \
  --seed 727300000 --games 256 --start 0 --chunks 3 --orbit 24 --temperature 1.5 --epsilon .04
"$py" "$run/fit_rollout_policy.py" "$work/validation" "$run/rollout_weights.npy" --validate
```

On 7,736 decisions with at least two legal cards, the fitted continuation matched the teacher's top action 74.82% of the time. The previous hand rule matched 52.75%. The mean teacher-logit gap to the top legal action fell from 4.56 to 0.93. The held-out game benchmark measures wins separately.

## Use a trained policy

`configs/pile_128_rollout.json` averages 128 possible deals per multi-action decision. It samples the remaining cards from the acting hand, public hand sizes and recent plays. Half the hypothetical tables use random opponents and half use the fitted strong continuation. It simulates up to 72 turns per deal and combines estimated win differences with the 24-suit-average teacher. Each player needs its own recurrent state.

```python
import torch
from public_lookahead import load_candidate

policy = load_candidate("live_training/improvement_20261118/lookahead/configs/pile_128_rollout.json", device="cpu")
state = policy.initial_state()

# decision is the native DecisionInput for this seat, including its legal mask.
with torch.inference_mode():
    result = policy.step(decision, state)
    action = result.logits.argmax().item()
    state = result.state
```

This short development run plays the candidate against RandomAgent and the pinned teacher at all three table sizes.

```bash
"$py" benchmark_multiplayer_winrate.py --candidate "$run/configs/pile_128_rollout.json" \
  --candidate-kind lookahead --seed 711800000 --blocks 24 --chunk-blocks 12 \
  --workers 1 --output "$work/smoke_win_rates.json"
```

To reproduce the three development-bank summaries from their saved per-deal outcomes, run `python live_training/improvement_20261118/lookahead/summarize_development.py`.
