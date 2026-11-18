# Reproduce the training critic

`leaf_weights.npy` contains the 1,300 fitted coefficients for the learned Pesten search value. The counterfactual rollout recipe uses these weights during training to estimate focal win probability after each candidate action. The public actor trained from those targets can play without a critic.

The training data contain 32,256 completed games, sampled at a probability of 0.18 on ordinary card-decision states. Each shard contains 384 game seeds. Shards 0000 through 0059 used the frozen single-view teacher. Shards 0060 through 0083 used four-view suit averaging. All used greedy legal actions for teacher seats and alternated between all-teacher tables and one teacher versus canonical random opponents. Player count was `2 + shard_index % 3`. The first deal in each shard used `1111000000 + 384 * shard_index`.

The collector stored own and opponent hands for computing training targets. Search deployment gives the value a hypothetical world sampled from public information and the focal hand.

## Fit from the archived trajectories

Run the following commands from the project root. The archive contains the original 84 input shards. Each shard was checked against its SHA-256 in `fit.json`.

```bash
leaf_dir=live_training/improvement_20261118/learned_search/fit2
leaf_work=$(mktemp -d)
puffer4-native/.venv/bin/python -m zipfile -e "$leaf_dir/training_shards.zip" "$leaf_work/shards"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  puffer4-native/.venv/bin/python train_public_leaf.py fit \
  --data "$leaf_work/shards" --out "$leaf_work/fitted" --chunks 84 \
  --epochs 28 --batch-size 2048 --seed 1111111707 --device cuda
sha256sum "$leaf_work/fitted/leaf_weights.npy"
```

A fresh extraction and full refit reproduced all 28 validation-history entries and the 10,528-byte coefficient file byte for byte. `reproduction.log` contains that run.

The trainer holds out shards whose index modulo 18 is 0, 1 or 2. This gave 277,322 training positions and 61,164 validation positions. The best validation epoch was 26, with winner log loss 1.012379 and focal-win Brier score 0.194635. The archived weight SHA-256 is `4cae698a302bb886c1304c12b2ba7c10cd88be63b3048cee90ea7d2cd15796fd`. The input archive SHA-256 is `5a687f0da91bb43d8e1a86a66f7e24c2133d6fe094df7a75753d424d1214eb0e`.

## Regenerate the trajectories

To regenerate the 84 shards from their seeds, point the collector at a fresh directory. `--eager-actor` reproduces the CPU actor used for the archived trajectories. The default collector uses a traced CPU actor. `collection_replay.log` verifies byte-identical regeneration of one single-view shard and one four-view shard.

```bash
leaf_generated=$(mktemp -d)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  puffer4-native/.venv/bin/python train_public_leaf.py collect \
  --data "$leaf_generated" --start 0 --chunks 60 --orbit 1 --games 384 \
  --keep 0.18 --seed 1111000000 --eager-actor
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  puffer4-native/.venv/bin/python train_public_leaf.py collect \
  --data "$leaf_generated" --start 60 --chunks 24 --orbit 4 --games 384 \
  --keep 0.18 --seed 1111000000 --eager-actor
```
