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
512-wide MinGRU. Cross-channel and new decoder weights start at zero. This is a
function-preserving widening: it avoids paying a large supervised relearning
bill merely to recover teacher behavior, while leaving 320 new channels per
layer available for RLVR.

The match gate is evaluated on held-out safe trajectories. It requires at
least 99.99% action agreement, absolute forward KL no greater than 1e-6, value
MSE no greater than 1e-8, and strategic-pass-rate difference no greater than
1e-4. A direct seat-balanced game non-inferiority test follows.

### Hawk RG-LRU

The alternative is a standalone 4-layer, 256-wide Hawk-style network with
320-channel RG-LRU state and gated MLP blocks. Its diagonal recurrent state is
linear in width and independent of context length. Training uses a parallel
signed log-cumulative scan; online decisions use the equivalent one-step
recurrence.

Distillation combines forward KL, reverse KL, hard teacher actions, value
matching, strategic-pass upweighting, and length-bucketed replay. DAgger rounds
aggregate fresh student-induced trajectories. Both a fixed teacher-policy
validation set and fresh student-policy validation set must reach 98.5% action
agreement and forward KL no greater than 0.04.

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
