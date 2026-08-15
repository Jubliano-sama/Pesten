# Teacher GRU gamma sweep

## Outcome

The strongest trained checkpoint used `gamma=0.99` and improved over the
untouched epoch-4387 teacher on the final fresh, seat-balanced comparison. The
weighted normalized win-share delta was **+0.02399**, with a paired-normal
99.9% interval of **[+0.00715, +0.04083]**. The corresponding raw weighted
score gain was **+0.00592**, or about 0.59 percentage points, with interval
**[+0.00041, +0.01143]**.

That does not establish that changing gamma caused the gain. On a separate
fresh comparison, the `gamma=0.99` checkpoint led the matched `gamma=0.97`
continuation by +0.01468 normalized win-share, but its 99.9% interval was
[-0.00322, +0.03258]. The supported conclusion is therefore that continued
teacher training produced a better research candidate, while the gamma choice
remains unconfirmed.

The candidate remains isolated and is not promoted:

```text
live_training/teacher_gru_gamma_sweep_20260814/gamma_0p99/training/latest.pt
SHA-256: fedd80e5cad46ece4ce20f20dcc7123a5099d25b62adee15840ae0c2e5c14ce9
epoch: 4447
```

## Matched training protocol

All four arms started from the exact promoted teacher checkpoint at epoch 4387
and used the same seed, 60 phases, 2,048 games per phase, player-count mix,
opponent cycle, learning rate, four PPO replay passes, fresh optimizer, and
7,920 optimizer steps. Each arm trained on 122,880 games; the total training
budget was 491,520 games.

The external privileged critic was disabled for every arm. Only gamma
0.9330329915368074 currently has a promoted matching external critic, so using
it for one arm would have mixed critic quality with the gamma effect. Every arm
instead used the actor's public value head. That head was originally fitted at
gamma 0.97, which is a conservative bias in favor of the historical control.

| gamma | terminal reward left after 20 decisions | screen delta | paired-normal 99.9% interval |
| ---: | ---: | ---: | ---: |
| 0.90 | 12.2% | -0.02817 | [-0.06226, +0.00591] |
| 0.9330329915 | 25.0% | -0.00582 | [-0.03872, +0.02708] |
| 0.97 | 54.4% | +0.01661 | [-0.01620, +0.04941] |
| 0.99 | 81.8% | +0.02828 | [-0.00497, +0.06152] |

No training arm stopped early. Every arm completed four PPO replay passes per
phase. Maximum approximate KL stayed below the 0.015 guardrail: 0.00257,
0.00245, 0.00298, and 0.01392 in ascending gamma order.

## Evaluation protocol

The screen used 300 matched deal blocks for every opponent/player-count
stratum. It covered 2-8 players, rotated the candidate through every seat, and
used three opponent lanes: the fixed starting teacher, the tracked MLP
champion, and canonical random. Each arm and the starting teacher played 31,500
screen games.

The screen winner was then evaluated twice with 1,200 blocks per stratum and
predeclared disjoint seeds:

- `gamma=0.99` versus the separately trained `gamma=0.97` continuation: +0.01468
  normalized win-share, interval [-0.00322, +0.03258].
- `gamma=0.99` versus the untouched epoch-4387 teacher: +0.02399 normalized
  win-share, interval [+0.00715, +0.04083].

Each model played 126,000 games in each confirmation. The baseline gain passed
the paired-normal 99.9% criterion but not the more conservative distribution-
free Hoeffding bound. The aggregate baseline gain was positive in all three
opponent lanes; only the MLP lane individually excluded zero at 99.9%.

## Decision

Use the gamma-0.99 checkpoint as the strongest research candidate from this
run, but do not change the training default or promote the checkpoint based on
this sweep alone. A fresh paired training-seed replication of gamma 0.97 and
0.99 is the clean next experiment for attributing the gain to reward decay
rather than continued-training stochasticity.

Machine-readable compact evidence is in
`benchmarks/teacher_gru_gamma_sweep_20260814.json`; complete commands, curves,
checkpoints, and raw evaluations are under
`live_training/teacher_gru_gamma_sweep_20260814/`.
