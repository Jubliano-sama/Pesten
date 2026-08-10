# PPO training performance — 2026-08-01

## Current result

Use the allocator-backed launcher for normal CLI training:

```bash
venv/bin/python run_ppo_fast.py \
  --iterations 1000 \
  --episodes 1000 \
  --eval_freq 10 \
  --eval_episodes 4000
```

The launcher only adds the verified system jemalloc preload and forwards every
argument to `ppo2.py`. Direct `ppo2.py` use remains supported.

One earlier warm, development-seeded RTX 3090 run of the accepted CPU-rollout
stack processed 1,024 games and three PPO updates in 1.186 seconds:

| Component | Seconds |
|---|---:|
| Rollout | 0.853 |
| GAE/update preparation | 0.030 |
| Three PPO updates | 0.304 |
| Total | 1.186 |

That is 1,201 games/s and 33,379 collected policy decisions/s end to end.
The GPU was observed idle immediately before and after the measurement. A
separate Puffer trainer was using about 2.7 CPU cores, so CPU-sensitive
measurements are explicitly treated as contention-bearing.

The final stack additionally routes policy-forward batches of at least 64 rows
through a strict-FP32 traced actor on the 3090, while game state, masks,
sampling, and the shrinking tail remain on CPU. In five fixed-state compiled
pairs this improved the complete rollout plus three PPO updates by 1.409x
(rollout alone: 1.636x). Its median was 0.678 seconds per 1,024 games, or
1,510 games/s. With the final native observation/mask kernel, a repeated
direct-Python production run completed 10,240 games in 6.372 seconds,
averaging 1,607 games/s. The exact recommended jemalloc launcher stack
completed the same seeded work in 6.030 seconds, averaging 1,698 games/s
(1.057x faster). Both runs produced the same warm and final actor/critic
hashes.

Combining the separately paired refinements gives a practical training-path
estimate of about 5.6x versus the inherited vectorized path. On mixed
compatible agent-set collection, the CPU-side collector improvements alone
remain about 8.34x. These are workload-specific figures, not a promise that
unrelated league schedules scale by the same factor.

## Scheduled evaluation

Evaluation still uses the canonical `Game` and original agent classes. Matches
are independently development-seeded and distributed across CPU processes;
the vector engine was not substituted because it produced winner mismatches.

For the CLI-default 4,000-game evaluation:

| Path | Seconds |
|---|---:|
| Optimized canonical sequential | 89.00 |
| Eight workers, including startup | 15.64 |
| Direct exact speedup | 5.69x |

Eight chunks per worker separately improved the default-scale worker run by
1.162x, giving a conservative combined estimate of about 6.6x for this
evaluation slice. Eight workers were faster than 4, 12, or 16 under the live
background load. Linux auto mode prefers CUDA-safe `forkserver`, with an
automatic `spawn` fallback for restricted environments.

No-collection evaluation also received three exact local improvements:

- fresh NumPy float32 feature construction exposed to Torch zero-copy;
- direct masking of the known play/suit action regions;
- direct normalized multinomial sampling without an unused log probability.

Feature tensors, sampled actions in the parity tests, turn counts, and winners
remain exact.

## Automatic compile policy

CUDA update compilation saves about 16.6% in steady update time but adds about
20.1 seconds of startup. Two paired 1,024-game probes put the break-even near
413,000 requested games. CLI auto mode therefore:

- enables compilation at 400,000 or more requested games;
- uses eager updates for shorter runs;
- always respects explicit `--compile_updates` or `--no-compile_updates`.

The default one-million-game command still compiles. Short experiments no
longer pay a compilation cost they cannot recover.

Compiled PPO updates now skip CUDA Graph capture for their changing minibatch
shapes while retaining the compiled CUDA kernels. Across two 10-iteration,
10,240-game pairs this reduced elapsed time from a 13.751-second median to
12.376 seconds (1.111x). The more conservative individual pair was 1.050x.
Actor and critic deviations from an eager reference remained at the same
relative-RMS scale as the previous compiled mode. This also removes Inductor's
warning after repeated dynamic shapes.

The current 1,024-game collector also uses six Torch threads. In a seven-seed
sweep, six was 1.064x faster than eight; in fixed-state RTX 3090 pairs it
improved the complete rollout plus three-update iteration by a median 1.075x
and the rollout slice by 1.099x. Observations, actions, rewards, episode
lengths, turn counts, and decision counts were exact. Retained log
probabilities differed by at most 1.91e-6, a normal reduction-order effect.
Six threads also leaves more CPU capacity for the concurrently running trainer.

Hybrid rollout inference keeps TF32 disabled only around the traced rollout
forward, then restores the training matmul setting. Across five parity seeds,
observations, actions, rewards, lengths, turns, and decision counts were exact;
retained log probabilities differed by at most 9.54e-7. TF32 rollout inference
was rejected because it changed trajectories. `--no-hybrid-rollout-inference`
provides an explicit CPU-only fallback.

## Accepted implementation areas

- traced/preallocated CPU rollout actor and vectorized observation/decision
  construction;
- native exact Smart policy, rollout transitions, and segmented GAE;
- native exact 121-feature observation and 55-action-mask construction;
- direct action sampling, NumPy masks, reusable workspaces and same-shaped
  engines;
- duplicate agent-set coalescing, compatible-set grouping, and true
  cross-engine coordination;
- direct single-engine dispatch, non-neural two-player draining, and a bounded
  historical-policy pool;
- CUDA actor/critic updates, TF32-high matmul policy, and compile break-even
  selection;
- strict-FP32 thresholded CUDA policy inference with CPU rollout state and
  sampling;
- jemalloc-backed launcher;
- canonical parallel evaluation and evaluation-only scalar-dispatch removal.

Cross-engine coordination now activates only when at least two engines exist.
This restores the established single-engine collector hook and improves the
common `smart,rl` workload by a paired median 1.119x while preserving rollout,
actor-state, and critic-state hashes.

## Rejected probes

The following were measured and left out because they were slower, below the
materiality threshold, or violated the numerical/game contract:

- full CUDA rollout under shrinking batches, OpenVINO/ONNX actor inference,
  inference freezing/fusion, TorchScript inference optimization, and
  persistent actor feature buffers;
- BF16/FP16 actor updates, static CUDA graphs, full-step/full-loss compilation,
  and max-autotune compilation;
- direct exponential-race replacement for `torch.multinomial`; all seven
  rollout hashes were exact, but the paired median gain was only 1.017x;
- vector-engine evaluation, which was fast but changed winners;
- persistent evaluation worker pools, evaluation observation-buffer reuse,
  local Torch generators, CPU-only seed shortcuts, CPU affinity pinning, and
  asynchronous evaluation (aggregate gain came with severe rollout stalls);
- jemalloc arena/decay tuning; two exact five-seed configurations were about
  3.1% slower than the allocator defaults and did not reduce peak memory;
- asynchronous checkpointing; synchronous temporary checkpoint writes were
  only about 28 ms for 40 MB.

## Verification

- Full suite: **526 passed, 5 skipped, 1,152 subtests passed**.
- Final focused rollout/evaluation/compile/launcher checks: 28 passed.
- Native observation/mask parity includes 2-, 3-, and 8-player layouts,
  forward/reverse play, retry state, and suit selection.
- `py_compile` and `git diff --check`: passed.
- Canonical parallel evaluation: per-game turns and winners exact.
- Training comparisons used development seeds only; no sealed outcomes were
  used.

Primary measurements are under `benchmarks/`, especially:

- `ppo_final_full_stack_20260801.json`
- `ppo_current_full_stack_20260801.json`
- `ppo_parallel_evaluation_default4000_20260801.json`
- `ppo_evaluation_chunksize_20260801.json`
- `ppo_compile_break_even_20260801.json`
- `ppo_dynamic_cudagraphs_20260801.json`
- `ppo_current_thread_sweep_20260801.json`
- `ppo_hybrid_rollout_inference_20260801.json`
- `ppo_native_observations_20260801.json`
- `ppo_single_engine_direct_dispatch_20260801.json`
