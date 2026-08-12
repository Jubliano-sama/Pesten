# SOTA distillation and RLVR campaign (2026-08-12)

## Objective

Produce two information-safe temporal candidates from the frozen epoch-3978
Puffer teacher, then compare matched pre/post RLVR changes under the same
2-8-player population. The serving constraint is sustained OPEX: long startup
and compilation are acceptable, but recurrent state, memory traffic, and
per-decision compute matter.

## Primary-source review

- [Policy Distillation](https://arxiv.org/abs/1511.06295) established dense
  distribution matching as a way to transfer an RL policy at expert-level
  performance.
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses sequential
  covariate shift by repeatedly labeling states induced by the student rather
  than training only on expert trajectories. This directly addresses the old
  large student's 90.1% in-dataset agreement but 40.3% head-to-head score.
- [Griffin / Hawk](https://arxiv.org/abs/2402.19427) combines stable real-gated
  linear recurrences, residual blocks, and gated MLPs. The recurrence has a
  fixed-size diagonal state; the paper reports lower long-sequence inference
  latency and higher throughput than attention baselines.
- [Mamba-2 / state-space duality](https://arxiv.org/abs/2405.21060) reports a
  2-8x faster selective-SSM core while remaining competitive with Transformers.
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464) combines forgetting gates
  and delta-rule memory and reports gains over Mamba-2 on retrieval and
  long-context tasks.
- [SPIRAL](https://arxiv.org/abs/2506.24119) supports continuously improving
  multi-agent self-play with game outcomes as verifiable rewards.

## Chosen models and methods

### 512x2 MinGRU

The teacher's 192 channels are embedded into the first 192 channels of a
512-wide MinGRU. Teacher channels cannot read from the new channels. The new
encoder and recurrent features retain random initial values, while their
decoder columns start at zero. This function-preserving widening avoids paying
a large supervised relearning bill merely to recover teacher behavior, and it
also lets decoder gradients activate the 320 new channels on the first RLVR
update.

The match gate is evaluated on held-out safe trajectories. It requires at
least 99.99% action agreement, absolute forward KL no greater than 1e-6, value
MSE no greater than 1e-8, and strategic-pass-rate difference no greater than
1e-4. A direct seat-balanced game non-inferiority test follows.

### Hawk RG-LRU residual

The accepted alternative retains the complete 192x2 teacher as a base branch
and adds a 2-layer, 128-wide Hawk-style residual branch with 160-channel RG-LRU
state and gated MLP blocks. Its diagonal recurrent state is linear in width and
independent of context length. The residual output heads start at zero, making
the complete student exactly teacher-equivalent while exposing nonzero adapter
features to the first RLVR update.

The first attempted alternative was a standalone 4-layer, 256-wide Hawk RG-LRU
trained with forward KL, reverse KL, hard teacher actions, value matching,
strategic-pass upweighting, length-bucketed replay, hard-example weighting, and
DAgger state aggregation. Replay agreement rose as high as 97.76%, but held-out
teacher-policy agreement plateaued around 89.7% and held-out KL worsened to
0.168. That branch was rejected as overfit instead of being promoted from its
training fit.

The residual design is a stronger distillation strategy for this setting: it
preserves the teacher function by construction and uses the different Hawk
recurrence only as learnable residual capacity. This meets the startup-cost-is-
acceptable and sustained-OPEX-first constraint without a Transformer cache or
a matrix-valued recurrent state.

## Pre-RLVR gate results

- Frozen teacher: MinGRU 192x2, 251,384 parameters, 384 recurrent-state
  elements.
- Trainable widened MinGRU: 512x2, 1,653,304 parameters, 1,024 state elements.
  On 224,445 held-out decisions it reached 100% action agreement, forward KL
  `3.16e-9`, and value MSE `1.66e-10`.
- Hawk residual: 695,024 parameters and 704 state elements. On 229,262 held-out
  decisions it had 100% agreement, zero KL, and zero value error.
- Each candidate then scored exactly 50.0% over 10,000 fresh, seat-balanced
  pairs against the teacher. The distribution-free 99.9% interval was
  [48.05%, 51.95%], clearing the declared 48% non-inferiority floor.

## Approaches not selected for this serving experiment

- A causal Transformer was not selected because its cache and repeated history
  processing grow with context, conflicting with the OPEX target.
- Full Gated DeltaNet was not selected for the first alternative because its
  matrix-valued recurrent state and specialized kernels are a worse fit for
  thousands of small simultaneous card-game states. The RG-LRU retains the
  modern selective-forgetting idea with linear state.
- Mamba-2 remains a strong future candidate, but a faithful implementation
  would add a separately versioned fused-kernel dependency. The current Hawk
  path is self-contained, auditable, and uses the existing privacy boundary.

## RLVR population contract

Every distilled model trains on exact terminal game outcomes against:

1. current-policy self-play;
2. the frozen latest warm MLP checkpoint;
3. canonical `randomAgent.AgentRandom` behavior.

The player-count cycle has 63 slots:

- 2, 3, and 4 players: 14 slots each (42/63 = 66.67%);
- 5 players: 6 slots;
- 6, 7, and 8 players: 5 slots each (21/63 = 33.33% tail).

Pre/post evaluation uses identical deal seeds, rotates the candidate through
every seat, and reports 99.9% paired normal and distribution-free Hoeffding
intervals. Improvement is a result only when the matched interval supports it;
training loss or an isolated development win rate is not a promotion claim.

## Capacity-order correction

The initial fixed-length three-model campaign was stopped after 119 completed
small-teacher phases (243,712 games) and before either larger model began RLVR.
The capacity order is now adaptive:

1. continue the 192x2 teacher in 60-epoch blocks;
2. compare every adjacent checkpoint on the same fixed-anchor, 21-stratum
   2-8-player pool;
3. call a monitor plateau only when the paired 99.9% upper confidence bound on
   normalized win-share improvement is at most 0.02 for two consecutive blocks;
4. confirm that window on a disjoint, twice-larger seed bank before stopping;
5. keep the 512x2 model frozen until that small-model plateau is confirmed.

The Hawk residual is the only capacity experiment allowed before then. It is
trained over the same phase seeds as the teacher so improvement per RLVR game
can be compared directly.

The plateau evaluator uses the exact `BatchedPesten` semantics and deterministic
batched recurrent inference. A cold two-block parity run matched the scalar
evaluator in every stratum and in both weighted deltas, while reducing elapsed
time from 10.064 seconds to 4.451 seconds (2.261x). This removes much of the CPU
cost of repeated high-confidence plateau checks without changing outcomes.
