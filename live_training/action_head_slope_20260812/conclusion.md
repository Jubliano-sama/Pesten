# 23-score action-head distillation and RLVR slope result

The 23-score MinGRU student matches the 55-logit teacher in game outcomes and
is modestly cheaper after compilation, but it does **not** show a better RLVR
improvement slope. Further RLVR on this branch is therefore stopped.

## Learned output contract

| Indices | Meaning |
| --- | --- |
| `0–3` | Playing-card suit scores: harten, ruiten, schoppen, klaveren |
| `4–15` | Ordinary rank scores: ace, 2…10, king, queen |
| `16` | Jack/boer score |
| `17` | Joker score |
| `18` | Pass score |
| `19–22` | Announced-suit scores after a jack |

The environment still receives its existing 55 concrete actions. Each of the
48 ordinary-card logits is the sum of one learned suit score and one learned
rank score; jack, joker, pass, and the four announced suits have dedicated
scores.

## Distillation match

The selected student has 245,208 parameters, versus 251,384 for the teacher
(2.4568% fewer). Exact action agreement is 92.7% because an additive suit/rank
head cannot reproduce every arbitrary 48-way teacher preference. The relevant
outcome gate did pass:

- 50,000 matched deals / 100,000 games against the teacher;
- student score: 50.4655%;
- paired normal 99.9% CI: 50.0111–50.9199%;
- distribution-free Hoeffding 99.9% CI: 49.5937–51.3373%;
- conservative 48% non-inferiority floor passed.

## Matched RLVR slope experiment

Both heads received 60 phases and 122,880 games with the same rollout seeds,
opponent schedule, player-count cycle, PPO settings, fresh optimizer, and GPU.
Checkpoints at phases 0, 13, 33, 53, and 60 were evaluated on common held-out
deals against the fixed teacher, MLP, and canonical random opponents. Player
counts 2/3/4 carry two-thirds of the mixture and 5–8 carry the one-third tail.

The normalized-win-share slope per phase was 0.0001675 for the full head and
0.0001609 for the factorized head. The factorized-minus-full slope was
−0.0000066 per phase, with paired normal 99.9% CI
−0.0007909 to +0.0007777. The factorized endpoint-improvement advantage after
60 phases was +0.01019, with 99.9% CI −0.03917 to +0.05955. Neither establishes
an improvement.

Opponent-specific factorized-minus-full normalized slopes per phase were:

| Opponent | Estimate | Paired normal 99.9% CI |
| --- | ---: | ---: |
| Canonical random | +0.0004922 | −0.0010231 to +0.0020076 |
| MLP | −0.0001060 | −0.0013665 to +0.0011546 |
| Fixed teacher/self anchor | −0.0004060 | −0.0016908 to +0.0008788 |

The random lane is a weak positive hint, but its interval is wide and both
other lanes point negative. No lane demonstrates a slope advantage.

## Endpoint and OPEX

At phase 60, a separate 50,000-deal / 100,000-game direct comparison scored
the factorized policy at 50.188% against the full policy. Its normal 99.9% CI
was 49.7226–50.6534%, and its Hoeffding interval was 49.3162–51.0598%: an
evidence-backed tie and a pass of the one-percentage-point non-inferiority
gate.

Excluding the first compilation phase, the factorized arm averaged 8.019
seconds per phase versus 8.134 seconds for the full arm (1.42% lower recurring
wall time), 383.7 versus 380.0 games/s, and 2.617 versus 2.678 update seconds.
Its first phase took 73.47 seconds versus 13.31 seconds, so the observed
compile premium breaks even after roughly 521 phases. Startup cost was declared
acceptable, but the recurring saving is still too small to compensate for an
unproven learning advantage on short campaigns.

## Selection

- Keep the plateau teacher as the performance champion.
- Keep the distilled 23-score checkpoint as the simpler, OPEX-efficient,
  outcome-matched deployment/training variant.
- Do not promote either 60-phase descendant as a stronger policy; the held-out
  evidence does not establish improvement.
