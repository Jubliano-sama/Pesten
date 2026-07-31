# Pesten benchmark dashboard

This directory owns the reusable benchmark normalization and dashboard
generator. It reads benchmark JSON without importing game, training, or agent
modules. The output is one canonical Data Analytics dashboard artifact plus a
portable HTML file produced by the packaged artifact builder.

## Generate

From the repository root:

```bash
python -m benchmark_dashboard.generate
```

The default input set is every top-level `benchmarks/*.json` file except
generated `artifact.json` and `normalized_runs.json`. Outputs:

- `benchmarks/dashboard/normalized_runs.json`
- `benchmarks/dashboard/benchmark_runs.sqlite`
- `benchmarks/dashboard/artifact.json`
- `benchmarks/dashboard/benchmark_dashboard.html` after packaging
- `benchmarks/dashboard/delivery-receipt.json` from the packaged builder

To package and verify the portable HTML using the canonical reader:

```bash
PLUGIN_ROOT=/home/tommie/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599
npm --prefix "$PLUGIN_ROOT" run report:deliver -- \
  --input /home/tommie/Pesten/benchmarks/dashboard/artifact.json \
  --output /home/tommie/Pesten/benchmarks/dashboard/benchmark_dashboard.html
```

The generator writes the normalized runs into SQLite and reloads them through
`SELECT payload_json FROM benchmark_runs ORDER BY run_id`; that actual query is
the canonical card/chart/table source while original JSON paths and SHA-256
values remain attached to every run. The delivery command’s JSON output is the
verification receipt. The HTML is generated output; do not edit it or add
another chart runtime.

## Future normalized inputs

New producers should emit either one run object or a `{ "runs": [...] }`
bundle matching
[`schema/benchmark-run.schema.json`](schema/benchmark-run.schema.json).
Use `null`, not zero, for metrics that were not measured.

Important semantics:

- A model role of `champion` or `candidate` must be explicit. The generator
  does not infer those roles from a filename, rating, or one matchup.
- `player_count` is 2, 3, 4, or null. Legacy tournament inputs infer 2 only
  when seat-0 and seat-1 game counts cover the full run.
- `environment_turns` and `policy_decisions` are separate. Legacy training
  `samples` map to collected policy decisions; environment turns stay null.
- Win-rate intervals are Wilson 95% intervals and are reconciled against
  wins/games.
- A candidate beats a declared champion with confidence only when the
  candidate interval is entirely above 50%.
- `status: incomplete` may omit performance metrics. Missing evidence remains
  a no-data state in the dashboard.
- Source path and SHA-256 are mandatory provenance. Checkpoint hash, recipe,
  observation version, benchmark git SHA, seed, and hardware remain null when
  not reported.

## Tests

```bash
python -m unittest discover -s benchmark_dashboard/tests -v
```

Tests cover legacy and normalized input handling, metric reconciliation,
malformed JSON and counts, and incomplete-run no-data behavior.
