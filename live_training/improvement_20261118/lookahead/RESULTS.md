# Public lookahead development results

The teacher with the fitted rollout policy won 50.22% averaged equally over six matchups. The frozen epoch-4387 teacher with 24-suit inference won 48.15% on the same deals. The paired difference was **+2.07 percentage points**, with an approximate 95% interval of **+0.53 to +3.61**.

## Development bank with 192 deals per matchup

The candidate took every seat once on each deal. All other seats played the listed opponent. Each policy played 3,456 games. The metric is wins divided by games, including draws in the denominator. The interval uses variation between paired seat-averaged deals within each of the six matchups. Each matchup has equal weight in the final average.

| Opponent | Players | Frozen teacher | Teacher with fitted rollout | Change |
|---|---:|---:|---:|---:|
| RandomAgent | 2 | 80.21% | 81.51% | +1.30 pp |
| RandomAgent | 3 | 60.24% | 60.24% | 0.00 pp |
| RandomAgent | 4 | 40.36% | 45.05% | +4.69 pp |
| Frozen teacher | 2 | 50.00% | 52.34% | +2.34 pp |
| Frozen teacher | 3 | 33.33% | 36.63% | +3.30 pp |
| Frozen teacher | 4 | 24.74% | 25.52% | +0.78 pp |
| **Equal-weight average** | | **48.15%** | **50.22%** | **+2.07 pp** |

The first deal seed was 735000000. Each successive matchup started 100000 seeds later, in the order shown. The common selection bank and the final tournament use separate deals.

## Changes to the search policy

| Policy at 128 sampled deals and 72 turns unless specified | Gain vs frozen teacher | Paired 95% interval |
|---|---:|---:|
| Retain the most recent 24 discards and use the hand-rule continuation | +0.74 pp | −0.84 to +2.32 |
| Estimate the current discard pile and use the hand-rule continuation | +1.14 pp | −0.45 to +2.72 |
| Same estimated pile and hand rule, with 224 sampled deals | +1.13 pp | −0.38 to +2.64 |
| Same estimated pile and hand rule, with the PPO93 actor | +1.40 pp | −0.28 to +3.09 |
| **Estimated pile and fitted continuation with the epoch-4387 actor** | **+2.07 pp** | **+0.53 to +3.61** |

The fitted continuation added +0.93 pp over the hand-rule continuation on these same deals (paired 95% interval −0.83 to +2.70). An earlier independent 120-deal bank used 80 sampled deals, a 32-turn horizon and a fitted leaf. Replacing its hand rule with the fitted continuation added +1.94 pp (−0.30 to +4.19). Each comparison holds all other policy settings fixed. In the earlier 128-deal bank at 728000000, introducing the hand-frequency model changed the fixed-pile search result by −0.09 pp (−2.33 to +2.16).

A separate audit compared estimated discards with actual hidden cards in 480 games at each table size. After at least 40 public plays, the fixed 24-card window excluded an average of 0.36, 0.77 and 1.00 cards that opponents actually held at 2, 3 and 4 players. Estimating pile recycling from public hand sizes and play counts reduced those errors to 0.03, 0.09 and 0.18. The hidden cards served as audit labels. Search receives only the acting hand and public observations.

The first 64 deals per matchup also tested a fitted policy for simulating strong players' suit declarations. It changed wins by −0.91 pp against the card-only fitted continuation (−3.86 to +2.04). The retained profile uses the original suit rule.

## PPO actor and search weighting

On 3,784 independent multi-card decisions, the PPO93 actor's centered legal logits had 0.56–0.57 times the scale of the epoch-4387 teacher. Its median top-two card logit gap was 51–55% of the teacher's. The measured configuration at search weight 72 therefore gives search more influence relative to the PPO93 logits. A weight of 38 provides a scale-matched comparison. The centered-logit scales come from 384 additional games on disjoint development seeds.

The first 64 deals per matchup on the 735M bank gave +3.12 pp for the epoch-4387 actor with fitted continuation, +4.21 pp for PPO93 with the same continuation and weight 72, and +2.39 pp for PPO93 with weight 38. The direct PPO93-minus-epoch-4387 difference at weight 72 was +1.09 pp (−1.70 to +3.87) on this subset.

Extending the same bank to 192 deals per matchup gave the following results. The direct PPO93-minus-epoch-4387 difference with the same search weight was **+0.85 pp** (−0.75 to +2.45). At weight 38 the PPO93 configuration was −0.36 pp (−1.95 to +1.23) relative to the epoch-4387 search configuration at weight 72.

| Fitted continuation and actor | RandomAgent 2p | 3p | 4p | Incumbent 2p | 3p | 4p | Gain vs frozen teacher | Paired 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch 4387, weight 72 | 81.51% | 60.24% | 45.05% | 52.34% | 36.63% | 25.52% | +2.07 pp | +0.53 to +3.61 |
| PPO93, weight 72 | 82.81% | 61.28% | 45.83% | 56.25% | 35.76% | 24.48% | +2.92 pp | +1.22 to +4.62 |
| PPO93, weight 38 | 82.29% | 58.68% | 44.79% | 55.99% | 33.16% | 24.22% | +1.71 pp | +0.07 to +3.35 |

The saved per-deal results and `summarize_development.py` reproduce the estimates. [Training commands and native usage](RUN.md)
