# Random-opponent ceiling-break run, 2026-08-01

## Decision

No candidate is promoted. None reached a reproducible 75% win/score rate on
fresh, seat-balanced games, and no candidate showed a positive paired
improvement whose 95% confidence interval excluded zero.

## Benchmark correction

The earlier 73-76% checkpoint peaks were underpowered selection results. The
top ten historical checkpoints were re-evaluated on 5,000 common unseen games
each. Their scores ranged from 68.58% to 69.24%; the best was:

`live_training/long_segments/segment_0003/mlp/ppo_actor_6.pth` at 69.24%.

Evidence:
`live_training/ceiling_break/checkpoint_confirm_top10_seed88750001.json`.

## New methods screened

| Method | Development result | Disjoint result | Decision |
|---|---:|---:|---|
| Exact suit projection and checkpoint ensembles | At most about 71% | No stable lift | Reject |
| Random-only PPO, three ordinary settings | At most 71.0% | Regressed | Reject |
| KL-regularized Monte Carlo policy gradient | 70.6% vs 70.0% parent on 500 games | No convincing lift | Reject |
| Fitted action-value residual | Selected peak 74.6% | 70.52% vs 69.46%; paired delta +1.06 points, 95% CI -0.28 to +2.40 | Keep as research signal only |
| Iterative fitted-Q | 72.2% selected peak | Lost the first-step signal | Reject |
| Evolutionary rank/pass/suit biases | 70.6% vs 69.85% validation | 69.68% vs 70.28%; paired delta -0.60 points | Reject |
| Deeper Bayesian information-set search | Best 68% in small screens | Below neural parent | Reject |
| 16-cycle large-batch PPO from confirmed parent | 70.3% selected peak | Actions effectively unchanged at low drift | Reject |

The strongest fitted-Q confirmation is:
`live_training/ceiling_break/fitted_q_a_confirm_g4_s64_seed88520001.json`.

## Training and rules constraints retained

- Seven cards per player.
- More decks are used as player count requires; seven and eight players use
  two decks.
- The main league recipe still covers two through eight players, with lower
  weight on the largest tables.
- The long-run restart interval remains 16 cycles.
- Learned policies receive only the established 121 public and own-hand
  features. Opponent card identities and draw-pile order are not exposed.

## Verification

- 29 focused tests passed.
- All new Python entry points compile.
- `git diff --check` passed.

