# Puffer GRU teacher strategy dashboard

This package measures the current 192-by-2 Puffer GRU teacher in paired
self-play deals and builds one canonical, read-only interactive interpretation
dashboard. Ordinary suits are intentionally combined by card number.

## Generate the measured data

Use the pinned PufferLib 4 environment from the repository root:

```bash
PYTHONPATH=. /home/tommie/Pesten/puffer4-native/.venv/bin/python \
  -m gru_strategy_dashboard.generate
```

The default run uses 512 deals for each table size from two through eight. It
runs each deal twice: once drawing from the model's own move weights and once
always taking its highest-scored legal move. Outputs:

- `benchmarks/gru_strategy/strategy_summary.json`
- `benchmarks/gru_strategy/artifact.json`

Rebuild the artifact without rerunning games:

```bash
PYTHONPATH=. /home/tommie/Pesten/puffer4-native/.venv/bin/python \
  -m gru_strategy_dashboard.generate --reuse-summary
```

## Package the interactive HTML

```bash
PLUGIN_ROOT=/home/tommie/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599
npm --prefix "$PLUGIN_ROOT" run report:deliver -- \
  --input "$PWD/benchmarks/gru_strategy/artifact.json" \
  --output "$PWD/benchmarks/gru_strategy/puffer_gru_teacher_strategy.html"
```

The packaged file is self-contained and keeps the supported filters, chart
tooltips, tables, and source details without a server or network connection.

## Metric meaning

- A number is *kept* when at least one card of that number is legal, but the
  teacher plays another number or passes.
- The memory comparison clears only earlier stored history. It retains public
  events observed since the teacher's previous turn.
- Separate matched checks clear the earlier memory stage, later memory stage,
  or all stored history and compare the favorite legal move.
- The current-turn view measures small changes around sampled decisions,
  adjusted by how much each input usually varied. It is not a global or causal
  importance score.
- The memory map compresses 384 stored values into two display directions;
  nearby dots are similar, but the directions do not have assigned meanings.
