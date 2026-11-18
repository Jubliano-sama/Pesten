# Outcome-trained search value

A 24-unit ReLU model estimates the winner from a sampled Pesten position. It scores each active seat and a draw result. Training uses full hands from completed games. At play the model receives only a hypothetical world generated from the acting seat’s hand, public plays and public hand sizes. The focal player never receives the opponents’ real cards.

`train_public_leaf.py` collected 23,040 games for `fit1`. `fit2` added 9,216 games with four-view suit-averaged actors. The corpus covers two, three and four players, and alternates strong self-play with a strong focal player versus canonical random opponents. Training holds out entire deal shards. `fit.json` in each model directory records seeds, row counts, validation history and hashes. `fit1` is the value used by `long.json` and `fit2` is the training critic for the separate rollout-improvement recipe.

## Predictions and play

On 48,887 held-out positions from the first corpus, the original hand-size heuristic scored 0.23816 on focal-win Brier loss and the learned model scored 0.19441. Winner log loss over the 48,135 decisive positions fell from 1.13845 to 0.95909. `leaf_cost.json` evaluates the actual seat-normalized heuristic on both metrics.

Development used matched deals and rotated all seats. Each cell measured outright wins over all games. The overall mean weights each player count and opponent equally. Intervals use paired deal-block differences. The frozen opponent was the production greedy teacher with 24-view suit averaging. The other opponent was the canonical random agent.

| Development profile | Deals per cell | Games per profile | Win-rate change vs frozen policy | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| H32, 80 samples, learned leaf, initial screen | 64 | 1,152 | +3.016 pp | +0.392 to +5.641 pp |
| H32, 80 samples, learned leaf, independent screen | 144 | 2,592 | +0.675 pp | −1.058 to +2.408 pp |
| H32, 80 samples, original leaf, same independent screen | 144 | 2,592 | +1.514 pp | −0.230 to +3.259 pp |
| H1, 96 samples, learned leaf, same independent screen | 144 | 2,592 | +0.367 pp | −1.101 to +1.834 pp |

Replacing only the original H32 leaf changed observed win rate by −0.839 pp on the independent screen, with a 95% interval of −2.200 to +0.522 pp. The interval leaves the incremental playing-strength benefit of the standalone learned leaf uncertain. `validation_comparisons.json` includes the cell-level counts and all paired comparisons. `pilot.jsonl` and `validation.jsonl` preserve the underlying block outcomes.

## Planner cost

Three warmed timing repeats used 288 real public positions with at least two playable cards, evenly split by player count. Median elapsed times per choice were 0.643 ms for H4/64 learned, 0.858 ms for H14/64 learned, 1.288 ms for H32/80 learned, 0.940 ms for H32/80 original and 2.124 ms for H80/128 original. This measures the compiled search on one CPU thread and excludes recurrent inference. The timing used uniform determinizations.

## Matched intervention training

`refine_leaf_contrasts.py` ran 27,648 pairs of complete games. The two arms shared their deal and random streams until a single public action changed. The collector captured the first completed afterstate, including terminal wins and timeouts, then compared estimates of the two arms to differences in actual focal wins. Of 16,537 eligible interventions, 13,779 trained the model and 2,758 validated it. The latter selected the starting checkpoint at mean squared error 0.32922, so the original `fit2` weights were retained. `fit_paired/fit.json` records every epoch.

## Additional development checks

A separate public style-classifier experiment used a 72-turn heuristic continuation for simulated strong opponents and a 3-turn learned continuation for simulated random opponents. On 128 independent deals per cell, 2,304 games per policy, the six-cell mean exceeded the frozen teacher by 1.693 pp (95% interval −0.136 to +3.522). The matching fixed-horizon heuristic exceeded the teacher by 1.476 pp. The paired difference between the two searches was +0.217 pp (−1.593 to +2.028 pp). These development-only settings were run in an experimental search loader.

The warmed, full-match timing repeated 24 deals per cell twice, for 864 games per policy. The frozen 24-view teacher ran at 11.41 games/s, H32/80 with the original leaf ran at 9.91 games/s, and H32/80 with the learned leaf ran at 8.90 games/s. The learned policy used 11.4% more elapsed time than the matching H32 heuristic. Each time includes all game simulation, queries for every player and the recurrent network. The shared machine load varied between repeats. `full_match_cost.json` records individual cell durations, turn counts and both repetitions.

A further 9,216 games used the full 24-view frozen teacher. `fit3` trained on 41,472 games in total. On 73,338 shared validation positions, `fit2` and `fit3` obtained winner log losses of 1.00714 and 1.00495, and focal Brier scores of 0.19251 and 0.19267. A separate categorical value tested at 1.04 μs/position compared with 2.61 μs for the dense value. Its validation log loss increased to 1.02556. `compact_check.json` contains each training cohort’s scores and the timed position counts.

On a held-out set of 2,758 actual paired interventions, 923 pairs changed focal win outcome. The original heuristic’s correlation with the realized change was −0.005. The correlations were 0.134 for `fit1` and 0.128 for `fit2`. Corresponding mean squared errors were 0.34051, 0.32874 and 0.32922. This check uses true engine afterstates as a training diagnostic. `contrast_validation.json` records the details.

The [fit2 reproduction guide](fit2/README.md) includes the original 84 training shards, how to refit the model, the frozen coefficient hash and how to regenerate the game data from seeds.
