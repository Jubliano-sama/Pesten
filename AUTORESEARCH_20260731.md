# Pesten player autoresearch — 2026-07-31

## Scope and selection rule

The player may use its own hand, the public game state, public action history,
and public hand sizes. It may not read opponent card identities or draw-pile
order. Training critics follow the same boundary.

The originally proposed sealed interval `[92_000_000, 93_000_000)` is retired:
the legacy unrestricted hash derivation placed development deal `92_286_746`
inside it, recorded in six aggressive-anchored screens. The active sealed
namespace is now `[4_294_000_000, 4_294_900_000)`. The repository-wide
`seed_integrity.py` scan of every currently recorded JSON/JSONL seed field
found zero active-namespace uses; updated random and hash-derived development
generators are bounded below `4_294_000_000`. The active namespace remains
reserved for one final evaluation after candidate and SHA-256 are frozen.
All sealed entrypoints share the fixed atomic claim ledger
`benchmarks/.sealed_evaluation_claim.json`. The ledger binds evaluator,
candidate SHA-256, configuration, seed block, and output before model loading
or play; its creation burns the holdout even if evaluation later crashes.

A pre-selection replacement had to improve the matched, seat-rotated
fixed-opponent pool without a material head-to-head regression against the raw
`ppo_actor_100run3.pth` starting policy. After the exact suit-projected player
was frozen, every later strength comparison used that projected production
policy as the incumbent; raw-vs-raw probes are diagnostic only. Screens choose
candidates and only fresh deal ranges are used for confirmation. The final
evaluation is protected by both the seed namespace and the atomic claim
ledger, so changing the output path cannot reopen the holdout.

## Starting raw incumbent

- Path: `ppo_actor_100run3.pth`
- SHA-256: `aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186`
- Architecture: 121-feature feed-forward PPO policy, 55 actions
- Production form after selection: exact 24-suit mean-logit projection,
  `alpha=1`, no player-label orbit

## Correctness repairs

- The batched draw-retry phase and corrected deployment wrapper expose only the
  newly drawn card and PASS. This preserves executable game semantics but is an
  intentional correction to the raw legacy actor observation, which exposed
  every compatible held card before the canonical game accepted only the newly
  drawn card.
- Canonical and batched engines reject fabricated, out-of-range, unowned, and
  incompatible card actions before mutating public state.
- Batched rollout supports the exact `SmarterAgent` policy.
- Temporal-to-PPO observation conversion now matches production semantics in
  ordinary, penalty, draw-retry, and Jack suit-choice states.
- Anchored PPO checks the pre-update importance ratio and fails before mutation
  when rollout/update behavior policies disagree.
- PPO GAE is episode-safe, rollout temperature is consistent, historical
  actors are frozen/cached, and frozen-champion assignment is configurable and
  audited.

## Broad development results

Rejected after matched-deal evaluation:

- naive PPO continuation and aggressive anchored PPO;
- tactical shielding and public-belief/Bayesian search;
- causal transformer, GRU/LSTM, tiny decoder, and SmolLM2 imitation adapters;
- a persistent `gpt-5.6-luna` player with cross-game context: it remained
  information-safe and legal, but lost 7-13 to the unprojected incumbent in
  its exploratory 20-game match;
- population distillation and a 10,000-game causal imitation run;
- confidence gates, temporal ensembles, and checkpoint interpolation;
- naive public-history auxiliary prediction;
- full player-label rotation/reflection averaging: its fresh 500-pair
  confirmation was neutral (`-0.0017`, CI `[-0.0207, +0.0177]`);
- the legacy draw-retry correction as a standalone strength wrapper (it
  remains a required executable-semantics correction);
- a corrected residual GRU: the first 1,000-pair direct screen was `+0.009`,
  but two fresh confirmations reduced the aggregate to `+0.0023`, CI
  `[-0.0039, +0.0086]`;
- information-safe Q-boosted PPO, potential shaping, champion-heavy PPO, and
  current-policy self-play. Current-policy iteration 3 briefly reached
  `+0.0093` pool delta on one common screen, but iteration 10 returned to
  `+0.0013`, and wrapping iteration 3 in the suit projection was outcome-level
  indistinguishable from wrapping the incumbent on the same fresh deals;
- compressed symmetry distillation. It was up to 21.9x faster than exact
  24-view inference at batch 128, but regressed head-to-head and had pool delta
  `-0.011` in its gameplay pilot;
- selective projection and lower-cost player-label subgroups. They are valid
  and technically useful, but did not clearly beat unconditional suit-only
  averaging on fresh deals;
- majority-vote and mean-Borda orbit aggregation. Borda passed its small
  screen, then regressed by `0.0111` pool score versus mean logits on its
  disjoint confirmation;
- a shared-symmetric two-player specialization of NashPG with the paper's
  current-to-reference KL direction and frozen-reference/outer-refresh
  schedule. A broad `alpha={0.05,0.2,0.5}` screen trained nine checkpoints.
  All fixed-pool screen intervals crossed zero and cross-play means stayed
  between `0.4972` and `0.5017`. The apparent `alpha=0.05`, outer-round-3
  leader then regressed on a fresh 200-pair confirmation: combined pool delta
  `-0.00917`, CI `[-0.01500,-0.00417]`, while its direct edge remained
  unresolved. A deeper `alpha=0.2` run recorded all 10,496 unique training
  deals and increased reference KL to roughly `1e-3`; its outer-round-4 screen
  looked promising, but a fresh 300-pair confirmation returned direct
  candidate advantage `-0.00333`, CI `[-0.04333,+0.03667]`, and pool delta
  `+0.00500`, CI `[-0.00611,+0.01611]`. It therefore also failed the
  predeclared two-gate rule;
- true TRPO with legal-mask `KL(old||new)` Fisher-vector products,
  conjugate-gradient natural-gradient steps, and rollback-safe line search.
  A common-data target-KL sweep over `{0.001,0.005,0.01}` produced no robust
  screen winner. The least aggressive candidate then lost its fresh 300-pair
  confirmation directly by `-0.02667`, CI `[-0.05000,-0.00667]`, with pool
  delta `-0.00333`, CI `[-0.01000,+0.00333]`;
- Data-Augmented Game Starts (DAGS) feasibility. Full simulator-private state
  capture/restore and the required root-versus-dataset observation flag were
  implemented and continuation/privacy-tested, but not connected to training.
  In the durable 256-game champion self-play audit, trajectories were already
  short (mean `42.35` turns, median `33`, p90 `84`, zero truncations). A safe DAGS pilot
  would require migrating every dense layer and collector from 121 to 122
  features and mixing root starts to avoid a biased equilibrium; its expected
  gain is lower than the iterative-reference lane on this environment;
- a development-only learned PPO response probe against the frozen raw actor.
  Its unanchored response appeared to gain `+0.160` directly in the 100-pair
  screen while losing `0.010` pool score, but that apparent exploit vanished
  on 300 fresh pairs: direct `+0.00333`, CI `[-0.0633,+0.0700]`, and pool
  `+0.00556`, CI `[-0.01944,+0.02944]`. Applying exact suit projection to both
  policies made the response weaker than production (`-0.040` direct,
  `-0.025` pool). This finite response therefore establishes neither a
  production exploit nor a stronger candidate;
- an exact differentiable 24-suit projected-PPO pilot. Rollout and update used
  the same projected actor, all initial ratios and KLs were exact, and
  full-batch target-KL overshoots rolled back the complete actor/critic/
  optimizer epoch. Its best two-update candidate screened at `+0.020` direct
  and `+0.00167` pool, with both confidence intervals touching or crossing
  zero. A deeper two-lane run then trained all 16 successive iterates on 4,608
  unique deals. Its 18,700-game trajectory screen found no two-gate winner:
  the best pool result was entropy iteration 3 at `+0.00667`, CI
  `[-0.00667,+0.02000]`, with direct delta exactly zero;
- top-advantage actor filtering and a minibatch parameter EMA. The broad
  one-iteration screen produced a boundary-touching full-data online result
  (`+0.010` direct and `+0.00667` pool), so fixed 25% and 100% recipes were
  deepened for eight iterations without retuning. The vectorized raw path
  processed 4,096 games in about 38 seconds. In the fresh exact-projected
  screen, all four deep online/EMA candidates had direct deltas between
  `-0.010` and zero and pool deltas between `-0.005` and `-0.00167`.
  Four precommitted post-hoc checkpoint EMAs over the projected-PPO
  trajectories also failed the pool gate. None was confirmed or promoted.

### Selected development candidate

The selected player is the exact frozen incumbent plus full group averaging
over all 24 relabelings of the four ordinary suits. Each transformed policy
output is mapped back to the original card/suit actions and the logits are
averaged before the normal legal-action mask and greedy action selection.
Special cards and all non-suit public/own-hand features remain fixed.

Across the three durable, disjoint confirmation ranges totaling 1,200 direct
paired deals and 3,600 paired opponent comparisons:

- direct advantage over the unprojected incumbent: `+0.05083`, bootstrap 95%
  CI `[+0.01333, +0.08750]`;
- combined pool delta: `+0.01556`, CI `[+0.00278, +0.02833]`;
- random delta: `+0.03083`, CI `[+0.00875, +0.05292]`;
- smart delta: `+0.00667`, CI `[-0.01625, +0.03000]`;
- smarter delta: `+0.00917`, CI `[-0.01167, +0.03000]`.

The largest individual 500-pair range was positive (`+0.02767`, CI
`[+0.00833, +0.04767]`); a second 500-pair range was neutral. Their combination
with the independent 200-pair confirmation remains positive, so selection does not rest
on one favorable seed block.

In seat-rotated multiplayer development screens against otherwise unprojected
champions, suit-only scored `0.3183` in three-player games (equal-policy null
`0.3333`, CI `[0.2833, 0.3533]`) and `0.2325` in four-player games (null
`0.25`, CI `[0.2088, 0.2563]`). These do not demonstrate a multiplayer gain,
but also do not establish a material regression. The more expensive
player-label orbit did not improve either result.

## Throughput

- Vectorized `BatchedPesten` rollout and evaluation replaced sequential
  per-agent inference for feed-forward PPO.
- Recurrent rollout uses one scoped CPU inference thread, measured at roughly
  10–20× faster than the oversubscribed path while preserving exact outcomes.
- A 14,000-game feed-forward matched benchmark completes at roughly
  176–204 games/s on the current host.
- Full suit averaging evaluates all 24 transforms as one tensor batch instead
  of making 24 game-agent calls; the wrapper also chunks base observations.
- The full player-label+suit orbit reached 226 games/s on CUDA in one
  2,800-game run, but was rejected for strength and portability reasons.

### Post-selection exact-semantics optimization

After the player and one-way result were frozen, only changes proven to retain
the same observations, actions, outcomes, and (for training paths) downstream
parameter updates were accepted:

- capping PyTorch CPU execution at the profiled eight-thread default reduced a
  representative 512-game PPO iteration from 43.43 seconds at 16 threads to
  17.55 seconds; a fresh post-lookup sweep across 4/6/8/10/12 threads again
  selected 8 (13.31 seconds median). The setting remains configurable and is
  recorded in recipes;
- immutable card/rank/compatibility lookup tables and one-pass Smart scoring
  reduced a 256-game rollout from 0.620 to 0.495 seconds (`1.253x`), with
  bit-exact rollout tensors and downstream SGD state;
- type-aware public-context slice writes reduced canonical 100-game execution
  from 6.390 to 6.008 seconds (`1.064x`) with exact observations, logits,
  actions, winners, and turn counts;
- tracing only the frozen CPU base actor at load time reduced representative
  single-decision 24-view projection from 0.776 to 0.506 milliseconds
  (`1.535x`), with bit-identical projected logits and masked actions; eager
  execution remains an explicit configuration opt-out;
- the opt-in CPU suit-inference chunk size 64 is neutral for batch 1, but
  reduces the largest estimated actor activation at batch 512 from 106.6 MB to
  13.3 MB and improved a 1,400-game development benchmark `1.40x`. The frozen
  production default remains 512;
- one-pass benchmark result indexing reduced synthetic assembly of the exact
  70,000-game result shape from 36.89 to 0.214 seconds (`172x`) while producing
  the same nested payload, and bootstrap resampling now preserves the exact
  historical RNG stream in bounded chunks;
- the generalized neural cross-play harness evaluated the nine-checkpoint
  NashPG screen plus incumbent over 15,000 matched, seat-rotated games in
  17.54 seconds (`855` games/s), including every neural pairing and common-deal
  random/Smart/Smarter comparisons;
- a 5,000-pair development layout now resolves its three hash collisions
  deterministically, preserving every non-colliding historical seed and
  yielding 20,000 distinct deals instead of aborting.

CPU compilation and gradient microbatching were rejected despite warm speed or
memory gains because they changed fixed-seed model state. Full production
compilation remains appropriate only on the already-supported CUDA path.

## Research basis

- Group averaging is the exact invariance projection used by the selected
  player; prior theory shows group-averaged predictors can reduce variance:
  <https://www.jmlr.org/papers/v21/20-163.html>.
- Partial augmentation can approximate full-group benefits but cannot enforce
  exact invariance in general, consistent with the subgroup results:
  <https://proceedings.mlr.press/v336/tahmasebi26a.html>.
- Large recent imperfect-information studies support generic policy gradients
  as strong baselines and explain why the corrected-PPO lanes were tested:
  <https://arxiv.org/abs/2502.08938> and
  <https://arxiv.org/abs/2605.28863>.
- The information-safe Q-boost experiment was inspired by
  <https://arxiv.org/abs/2605.19235>, but deliberately did not use that paper's
  centralized privileged critic.
- A July 2026 Gin Rummy study independently found that larger attention,
  recurrent, and live-LLM agents did not break its information ceiling, while
  trust-region updates, curricula, warm starts, and checkpoint selection were
  useful. That result supports testing all of those lanes here without
  assuming that capacity alone would win:
  <https://arxiv.org/abs/2607.06854>.
- The iterative-reference experiment implements the practical objective and
  reference schedule in NashPG, as a shared-policy specialization rather than
  claiming literal separate-player parameterization:
  <https://arxiv.org/abs/2510.18183v2>.
- The independent trust-region lane implements the constrained natural-gradient
  update and backtracking acceptance procedure from TRPO:
  <https://arxiv.org/abs/1502.05477v5>.
- DAGS motivated the audited intermediate-state scaffold, while its own bias
  counterexample and task-flag mitigation are why it was not wired into the
  frozen 121-feature player without a separate controlled study:
  <https://arxiv.org/abs/2605.14379>.
- The learned-response diagnostic follows PPO's clipped policy objective and
  the response-oracle framing used by PSRO, while deliberately making no exact
  exploitability claim: <https://arxiv.org/abs/1707.06347> and
  <https://arxiv.org/abs/1711.00832>.
- The top-advantage/EMA lane transfers two ablated ingredients from the 2026
  Generals.io self-play study—filtering actor updates to the highest-advantage
  samples and evaluating an exponential moving average—without transferring
  its privileged environment state: <https://arxiv.org/abs/2606.23348>.

## Final selection

Frozen before sealed evaluation:

- checkpoint: `ppo_actor_100run3.pth`;
- checkpoint SHA-256:
  `aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186`;
- projection: 24 ordinary-suit permutations, mean logits, alpha `1.0`;
- no player-label orbit and no adaptive selector;
- greedy masked action selection with exact draw-retry semantics;
- production entrypoint: `best_player.py`;
- interactive entrypoint: `python interactive_game.py --agents human,best`.

The one-way sealed holdout used 5,000 paired deals per matchup: 5,000 direct
pairs plus 5,000 pairs against each of random, smart, and smarter, for 70,000
games total. It independently confirmed the candidate:

- direct candidate score: `0.5203`, CI `[0.5113, 0.5294]`;
- direct advantage: `+0.0406`, CI `[+0.0226, +0.0586]`;
- combined pool delta: `+0.01678`, CI `[+0.01050, +0.02300]`;
- random delta: `+0.02270`, CI `[+0.01160, +0.03400]`;
- smart delta: `+0.02170`, CI `[+0.01020, +0.03300]`;
- smarter delta: `+0.00595`, CI `[-0.00420, +0.01635]`.

The 70,000 games completed in 723.8 seconds at 96.7 games/s. The raw sealed
result SHA-256 is
`3b1b634baa39f14d2d32fe081f124bb9a11175b96614fa578a8d7684ef06f688`;
the atomic claim-ledger SHA-256 is
`21d6f27508e29b97adc8708d7d76f2a2a5dadf6b67469ed34dde7073b3cb03eb`.

Final verification after the additional NashPG, DAGS, and TRPO lanes: 388 tests
passed, 4 skipped, and all 1,152 equivalence subtests passed. Active privacy
tripwires covered 216 decisions across 2–7 players, every absolute seat, and
ordinary, penalty, suit-change, and draw-retry phases: opponent-card and pile
access remained zero, every action was legal, and paired hidden worlds chose
identical actions. Compilation, diff checks, exact checkpoint verification,
canonical/batched parity, the recorded-artifact-only release verifier, and
sealed-protocol guards passed.
