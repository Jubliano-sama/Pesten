# Counterfactual rollout policy

`belief_model1/branch_195.pt` is a 91 KB action head and public opponent classifier for the 24-view suit-averaged teacher at epoch 4387. Inference uses the acting hand, public play and draw events, public hand sizes, and the teacher's scores. The checkpoint embeds the hand-frequency table and style classifier. The full-state critic is used during training only.

## Play or benchmark

Run these commands at the repository root in the native Puffer environment. The recorded environment used Python 3.12.12, PyTorch 2.13.0+cu130, PufferLib 4.0.0, NumPy 2.5.1 and the optional Numba 0.68.1 with llvmlite 0.50.0 for CPU simulation. The actor and rollout collection run on CPU. The fitting commands below used an RTX 3090.

```bash
PY=puffer4-native/.venv/bin/python
ART=live_training/improvement_20261118
WORK=$(mktemp -d)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv pip install --python "$PY" --target "$WORK/jit" --no-deps numba==0.68.1 llvmlite==0.50.0
export PYTHONPATH="$WORK/jit${PYTHONPATH:+:$PYTHONPATH}"

"$PY" benchmark_multiplayer_winrate.py --candidate-kind residual \
  --candidate "$ART/recipe/belief_model1/branch_195.pt" \
  --seed 519250000 --blocks 128 --chunk-blocks 64 --workers 2 --output "$WORK/wins.json"
```

A normal game uses the same actor and the game's ordinary legal-action handling, including draw retries.

```python
from game import Game
from randomAgent import AgentRandom
from temporal_puffer_agent import temporal_agent_factory
from train_outcome_residual import load_candidate

policy = load_candidate("live_training/improvement_20261118/recipe/belief_model1/branch_195.pt")
player = temporal_agent_factory(policy, deterministic=True)
turns, winner = Game([player, AgentRandom, AgentRandom]).auto_simulate()
```

## Rebuild the training inputs

The first dataset is shared between the style classifier and hand-frequency estimates. Generate all 108 shards in the same directory. Each contains 384 games split between greedy teacher opponents and canonical RandomAgent opponents, across the three table sizes and all seats. The focal policy samples from softened teacher logits and a uniform legal-action mixture.

```bash
"$PY" train_outcome_residual.py collect --directory "$WORK/exploration" --games 384 --seed 510000000 \
  --start 0 --chunks 72 --orbit 1 --temperature 2 --epsilon .04
"$PY" train_outcome_residual.py collect --directory "$WORK/exploration" --games 384 --seed 510000000 \
  --start 72 --chunks 36 --orbit 4 --temperature 1.65 --epsilon .025
"$PY" opponent_style.py --directory "$WORK/exploration" --output "$WORK/style" \
  --device cuda --epochs 20 --seed 631229811
"$PY" train_public_lookahead.py "$WORK/exploration" "$WORK/hand_frequencies.json" \
  --random-games 1536 --seed 726000000
```

The critic fits the eventual winner on 338,486 sampled states from 32,256 complete games, with entire deal shards held out. The first 60 shards use the raw teacher. The other 24 use four suit views. The fitted model selected epoch 26 by held-out winner log loss. The recorded critic can also be loaded directly from `learned_search/fit2/leaf_weights.npy`.

All 84 original critic-training shards are archived with their hashes. [Critic replay instructions](../learned_search/fit2/README.md)

```bash
"$PY" train_public_leaf.py collect --data "$WORK/critic-data" --seed 1111000000 --games 384 \
  --start 0 --chunks 60 --orbit 1 --keep .18
"$PY" train_public_leaf.py collect --data "$WORK/critic-data" --seed 1111000000 --games 384 \
  --start 60 --chunks 24 --orbit 4 --keep .18
"$PY" train_public_leaf.py fit --data "$WORK/critic-data" --out "$WORK/critic" \
  --chunks 84 --epochs 28 --batch-size 2048 --device cuda --seed 1111111707
```

## Train the action head

At eligible focal turns the simulator clones the complete game, each seat's decision memory and the game’s random streams. Each branch forces a different legal move within eight logits of the greedy action. All branches follow greedy play or canonical RandomAgent thereafter. Labels average estimated focal win probability after one, four and twelve completed turns with weights 0.12, 0.25 and 0.63. Finished games use the actual binary win result. The actor learns the within-position difference after subtracting the teacher's fitted contribution of 0.00263 win probability per logit. Training weights the six table-size and opponent combinations equally.

```bash
"$PY" rollout_policy_improvement.py collect --directory "$WORK/forks" --leaf "$WORK/critic/leaf_weights.npy" \
  --seed 516200000 --start 0 --chunks 36 --games 192 --orbit 1 --keep .65 --explore .2 \
  --margin 8 --horizons 1 4 12 --beliefs
"$PY" rollout_policy_improvement.py fit --directory "$WORK/forks" --output "$WORK/actor" \
  --chunks 36 --device cuda --style "$WORK/style/style.pt" --beliefs "$WORK/hand_frequencies.json" \
  --graph --epochs 16 --batch 512 --learning-rate .0015 --weight-temperature 1.5 \
  --fit-seed 512750000 --horizon-weights .12 .25 .63 --margin 8 --scales 35 95 195
```

The run produced 95,643 action branches at 40,450 positions from 6,912 games. Validation holds out whole shards whose index modulo 15 is 0, 1 or 2. `belief_model1/inputs.json` records the dependencies and exact shard hashes. Recollecting shard 0 reproduced every array and the compressed file byte for byte. `belief_model1/fit.json` records the held-out action-value error and how often each scale changed the predicted move. The actor mixes its two action heads using the style probability inferred from public observations.

On the 128-deal-per-cell development screen, scale 195 averaged 49.13% outright wins against the six target configurations, compared with 47.62% for the incumbent on the same games. The paired 95% interval for the 1.51 percentage-point change was −0.18 to +3.20. Scale 95 gained 0.84 points. These data selected the candidate. The independent evaluation is reported separately. `dev_belief1.jsonl` contains the paired results and `belief_model1/scalar_parity.json` records agreement with the normal `Game` on game outcomes and turn totals for one additional seat-rotated deal per cell.

A second development bank used 112 new deals per cell. The original scale-195 actor gained 1.30 percentage points over the incumbent, with a paired 95% interval from −0.71 to +3.31. A larger model trained on 240,243 branches from 18,432 games changed average win rate by −0.35 and −0.38 points at scales 175 and 300. The original scale-195 actor advanced to independent selection. [Paired second-bank results](dev_belief2.summary.json)

## Training speed

The complete nominated run spent 282.9 seconds collecting the 36 shards. The 16-epoch GPU fit took about 40 seconds. Both figures use the pretrained critic, style classifier and hand frequencies.

A separate matched CPU benchmark trained the same architecture on three independent seeds, with six 96-game shards and all 16 fitting epochs per seed. Each arm collected 1,728 games and 24,402 forced moves, compressed its shards, trained the action head and wrote the checkpoint. Process wall time includes startup. Both arms used one pinned core, the same prewarmed numerical cache and the same auxiliary checkpoints. The old arm restores the pre-optimization rollout modules and uses the eager frozen teacher. The new arm uses batched state extraction, sparse event encoding and the traced teacher.

| Three matched runs combined | Original | Optimized |
| --- | ---: | ---: |
| Collection | 116.83 s | 98.04 s |
| CPU fit | 9.83 s | 10.60 s |
| End-to-end wall time | 132.52 s | 114.31 s |
| Games per second | 13.04 | 15.12 |

End-to-end training throughput increased 1.159×. Individual speedups were 1.163×, 1.176× and 1.139×. All 18 compressed training shards, all validation loss histories and all three final policy checkpoints matched exactly. [Paired timings and output hashes](branch_training_timing.json)

```bash
"$PY" "$ART/recipe/benchmark_branch_training.py" --cpu 2 \
  --scratch "$WORK/speed" --output "$WORK/branch-speed.json"
```
