# Dashboard design contract

## Brief

- Audience: developers comparing Pesten agents and training recipes.
- Decision: identify a declared champion, test a candidate against it with
  uncertainty, quantify evaluation/training cost, locate runtime bottlenecks,
  and judge reproducibility and information safety.
- Surface: canonical Data Analytics dashboard artifact packaged as one
  self-contained, read-only HTML file.
- Snapshot: reviewed JSON files under `benchmarks/`; no live connector.
- Global filter: player count, with explicit 2-, 3-, and 4-player values when
  source data exists.

## Metric roles

- Hero: declared champion, candidate verdict, largest evaluation wall time and
  environment turns, final-recipe policy decisions/s, final 100-episode time.
- Diagnostic: matchup win rates, Wilson intervals, cross-play cells,
  Bradley–Terry ratings, seat bias, seed variance, throughput, and phase time.
- Guardrail: correctness observations, illegal actions, information leakage,
  resource coverage, and missing-source states.
- Detail: normalized run metrics and provenance fields.

## Chart map

| Section | Question | Family / native type | Fields | Supported takeaway | Palette |
| --- | --- | --- | --- | --- | --- |
| Tournament evidence | How did each named agent perform? | Comparison / `horizontalBar` | matchup, win_rate, games, CI | Point estimates only; intervals are in the table | Single blue root, neutral 50% line |
| Cross-play | Which observed pairs favor which side? | Matrix / native table | model rows, opponent columns, win_rate | Observed cells; missing pairs stay blank | Exact lookup; no custom chart runtime |
| Ratings | What order does the observed schedule imply? | Ranking / `horizontalBar` | model, rating, connected_games | Descriptive only because the schedule is incomplete | Single blue root |
| Seat bias | Does seat affect the named agent? | Signed comparison / `horizontalBar` | matchup, seat_bias_pp | Direction and percentage-point magnitude around zero | Blue/orange diverging, neutral zero |
| Training throughput | Which configuration collects decisions fastest? | Ranking / `horizontalBar` | configuration, policy_decisions_per_second | Source-backed speed with stage/status context | Single blue root |
| Seed variance | How variable is final throughput? | Discrete comparison / `bar` | seed, policy_decisions_per_second | Three separate seed observations, not a fitted distribution | Single blue root |
| Time spent | Where is final wall time spent? | Composition / `stackedBar` | seed, phase, seconds | Rollout versus non-rollout composition | Blue/orange categorical |

## No-data policy

Missing champion roles, learning curves, 3-/4-player tournaments, CPU/GPU
utilization, illegal-action results, information-leakage results, observation
versions, and benchmark git SHAs are rendered as explicit no-data or partial
states. They are never synthesized, zero-filled, or promoted to pass/fail.

Repeated matchup files remain separate in the evidence table. Cross-play and
Bradley–Terry views select the largest-game run for each unordered pair so
possibly overlapping runs are not silently double-counted.
