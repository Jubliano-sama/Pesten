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
