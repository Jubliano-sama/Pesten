# Best model artifacts

This directory is the Git-tracked home for model checkpoints that pass the
project's declared promotion gate. Raw, intermediate, and merely promising
training checkpoints remain ignored.

For every promoted model, add:

- the exact `.pt` or `.pth` checkpoint;
- a compact JSON result or manifest with its SHA-256, evaluation seeds, gate
  thresholds, and final results; and
- the report or code change that identifies the checkpoint as the selected
  model.

Use a stable descriptive filename rather than `latest.pt`, then stage the
checkpoint and its evidence together. A model is not "best" based only on
training loss, an unfinished run, or a development-set result.

## Current promoted models

- `puffer_min_gru_192x2_plateau_teacher_epoch4387.pt`: confirmed small-model
  plateau and performance champion.
- `puffer_min_gru_192x2_factorized23_distilled_epoch4387.pt`: outcome-matched
  23-score variant with lower parameter count and modestly lower recurring
  training OPEX.
- `ppo_omniscient_mlp_gamma0p995_2to8_20260815.pt`: gamma-retargeted causal
  omniscient 1.07M-parameter PPO value estimator trained across 2-8 players.
  It is the default external critic in `train_puffer_fast.py` and matches the
  default `gamma=0.995` reward contract.
- `ppo_omniscient_mlp_gamma025at20_2to8_20260814.pt`: the earlier causal
  omniscient critic, retained as the audited hot-start source and for
  reproducibility of the gamma-0.9330329915 reward contract.

Their hashes, gates, evaluation seeds, output contract, and selection decision
are recorded in `puffer_min_gru_192x2_20260812_manifest.json`,
`ppo_omniscient_mlp_gamma0p995_2to8_20260815_manifest.json`, and
`ppo_omniscient_mlp_gamma025at20_2to8_20260814_manifest.json`.
