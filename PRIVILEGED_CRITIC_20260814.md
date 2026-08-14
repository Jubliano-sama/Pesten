# Causal omniscient PPO critic investigation

This investigation replaces PPO's public-observation value baseline with a
separate environment-side critic. The actor contract is unchanged. At each
decision, the environment captures a snapshot before the action is sampled or
applied, evaluates the external critic, and stores that value only for GAE.

## Promoted default follow-up

The production follow-up changes PPO's default discount to
`gamma=0.9330329915368074`, for which `gamma**20 = 0.25`. Thus a terminal
reward component depreciates by 6.6967% per controlled-seat discount step and
retains exactly 25% after 20 such steps. The separate `-0.01` per-decision
reward is unchanged and is accumulated into the return in addition to the
discounted terminal component.

Following the user's final model choice, the selected count/moment MLP was
trained for all 24 scheduled epochs on the 2-8-player corpus. Validation picked
epoch 7 and rolled back later overfitting. The resulting model has 1,070,291
parameters and uses a 256-wide, two-layer residual MLP with 96-wide structured
feature groups. Its calibrated validation MSE was 0.161585, versus 0.203858 for
the inherited public head.

The checkpoint and hyperparameters were then frozen before a new confirmation
split was generated: 1,792 games, 3,248 controlled-seat trajectories, 72,060
decisions, all player counts 2-8, and zero deal-seed overlap with train,
validation, or the historical test split. The historical test split received
only structural and overlap audits in this follow-up.

| Estimator on fresh confirmation | MSE | Explained variance | Correlation |
| --- | ---: | ---: | ---: |
| inherited public head | 0.206105 | 0.194734 | 0.480833 |
| public head calibrated on validation | 0.176237 | 0.227714 | 0.477195 |
| promoted omniscient MLP | **0.170424** | **0.253235** | **0.503251** |

Against the raw public head, the paired deal-clustered MSE delta was -0.035680
(95% CI [-0.042180, -0.029034]). Against the stronger validation-calibrated
public comparator it was -0.005812 (95% CI [-0.007439, -0.004243]). This fresh
confirmation result, rather than training loss or a previously opened split,
is the promotion basis.

The exact promoted checkpoint is:

```text
models/best/ppo_omniscient_mlp_gamma025at20_2to8_20260814.pt
SHA-256: 63fc1b610da5fa40838a96bb3157cb1aab328ac1a46ce1a3c823af484e0b9d6d
```

`train_puffer_fast.py` now loads it by default. This is a value-estimator
promotion; it does not by itself establish that a newly trained actor is
stronger.

The remainder of this report records the earlier `gamma=0.97` architecture and
information study. Its MSE values have a different target distribution and are
not numerically comparable with the promoted follow-up above.

## Causal boundary

The critic may see any fact that exists at the current turn:

- every player's cards, card multiplicities, hand length, and engine hand order;
- compatible recurrent memories for every neural-controlled seat;
- draw-pile composition and current top-first order;
- discard composition and current top-first order;
- cumulative played-card counts and the 64 most recent plays;
- phase, suit, rank, direction, penalty, player/deck count, seat, turn count,
  hand-size summaries, and opponent role;
- the acting policy's current action probabilities, recurrent memory, and
  inherited public value estimate.

It never receives the selected action, reward, winner, a future event, deal
seed, or RNG state. The RNG exclusion is deliberate: even though RNG state is
physically present, it would let the critic decode future samples instead of
estimating their conditional expectation. All snapshot arrays are immutable.
Replay audits reconstruct each trajectory and verify the initial deal, legal
masks, public events, winner, and turn count.

## Original gamma=0.97 evidence protocol

The fixed actor was
`puffer_min_gru_192x2_plateau_teacher_epoch4387.pt`. Data cover 2-8 players and
five opponent lanes: current self-play, the PPO champion, Smart, Smarter, and a
historical temporal policy.

| Split | Games | Trajectories | Decisions | Unique deal seeds |
| --- | ---: | ---: | ---: | ---: |
| Train | 28,672 | 51,632 | 1,105,497 | 28,672 |
| Validation | 896 | 1,624 | 37,046 | 896 |
| Test | 1,792 | 3,248 | 69,946 | 1,792 |

The three deal-seed sets are disjoint. Card conservation, hand-order/count
agreement, pile length/composition, policy-probability, recurrent-memory, and
finite-return audits all report zero violations. Architecture, information,
capacity, auxiliary-loss, seed, calibration, and ensemble choices used only
train and validation. The test split was opened once after the finalist was
locked.

The target is the same discounted terminal-plus-step return PPO uses
(`gamma=0.97`). MSE is therefore the primary value-estimation metric. Paired
confidence intervals resample independent deal seeds, keeping all seats and
trajectories from a deal together.

## Original gamma=0.97 finalist

The selected member is a 1,070,291-parameter structured critic:

- shared per-player encoding followed by two layers of four-head attention,
  padded to 13 players;
- positional-moment encoding of full hand order;
- GRU encoders for ordered card sequences;
- 96-wide feature-group encoders and a 256-wide residual MLP value backbone;
- direct residual initialization from the inherited public value;
- auxiliary win and remaining-decision objectives; and
- a validation-fitted affine correction for each player count.

Two independently initialized members are averaged and then calibrated,
yielding 2,140,582 parameters. The self-contained local finalist is:

```text
live_training/privileged_critic_20260813/models/
  selected_mlp_h256_g96_ensemble2_final/ensemble/best.pt
SHA-256: 241b7db3df231ad6d2efa29713d97add9af931417e8dcb52a2d59649d64c495e
```

The checkpoint is intentionally under ignored experiment storage, not promoted
as a production playing model.

### Why not an outer GRU?

At matched small capacity and without calibration, the full-state MLP won:

| Backbone | Parameters | Validation MSE |
| --- | ---: | ---: |
| MLP | 315,035 | 0.257614 |
| causal Transformer | 776,475 | 0.258014 |
| LSTM | 513,051 | 0.258824 |
| GRU | 447,003 | 0.259063 |
| public-only GRU | 447,003 | 0.260709 |

The current physical state plus each neural seat's current policy memory is
already a near-Markov representation. An additional turn-level recurrence
mostly relearns information present in the snapshot and was harder to optimize.
The implementation still supports GRU, LSTM, and causal Transformer backbones,
and the selected model still uses recurrence where order matters: draw and
played-card sequences. Player attention, rather than a fixed flat player vector,
lets capacity scale cleanly with player count.

### Sequential hand-token experiment

The requested hand-token variants were also tested under the original
`gamma=0.97` target. They retained every card and either embedded exact card IDs
or used a lossless rank-plus-suit factorization before a hand GRU. The outer
value backbone remained an MLP so this isolated the hand representation.

| Hand representation | Calibrated validation MSE |
| --- | ---: |
| prior best count/moment MLP | 0.251470 |
| longer exact count/moment MLP in the matched run | 0.252414 |
| exact-card hand GRU | 0.253016 |
| rank+suit hand GRU | **0.250023** |

The rank+suit hand GRU improved 0.001447 MSE, or about 0.58%, over the prior
best single MLP on that validation target. A bidirectional variant reached
0.255315 before calibration at epoch 6, but its run was stopped when the model
direction changed to the longer-trained MLP. None of these old-target token
checkpoints was promoted, and their values must not be compared directly with
the new-discount MSEs above.

A larger 2,767,523-parameter member did not help (calibrated validation MSE
0.251613 versus 0.251470 for the selected member). Removing the auxiliary
objectives was clearly worse (0.254162). A third initialization failed to
improve beyond the inherited head and was excluded before test evaluation.

## Which current-turn information mattered

These controlled 128-wide MLP ablations were intentionally small screens.
Lower MSE is better; differences are conditional on all other admitted fields,
so they should not be read as causal effects or added together.

| Removed group | Validation MSE | Reading |
| --- | ---: | --- |
| none (full) | 0.257742 | reference |
| inherited public value | 0.259242 | strongest single prior |
| opponent recurrent memories | 0.258585 | useful hidden policy state |
| hand order | 0.258566 | useful beyond card counts |
| all omniscient hands | 0.258532 | useful hidden physical state |
| played-card history | 0.257980 | small useful signal |
| draw information | 0.257944 | small useful signal |
| player roles | 0.257889 | small useful signal |
| engine controls | 0.257747 | redundant with other fields in this screen |
| current policy context | 0.257688 | mostly redundant with public-value residual |
| discard information | 0.257672 | composition retained for future recycle semantics |
| duplicated safe observation | 0.257615 | safely removed from the selected critic |

For the undrawn pile, composition alone scored 0.257996. Adding the next card
improved it to 0.257809; the next four was best at 0.257691. Depths 8, 16, and
32 were statistically/operationally flat at 0.257714-0.257734, so the selected
input keeps composition plus the top four cards. Full discard order is omitted:
after recycle the engine shuffles it, while discard composition remains
physically relevant.

## Original gamma=0.97 held-out result

| Estimator | Test MSE | Explained variance | Correlation |
| --- | ---: | ---: | ---: |
| inherited public head | 0.286850 | 0.231798 | 0.484577 |
| privileged two-member ensemble | **0.284365** | **0.238623** | **0.488558** |

The MSE delta is -0.002486, or a 0.87% reduction. Its paired deal-clustered
95% confidence interval is [-0.004723, -0.000208]. A validation-calibrated
version of the public head shifted poorly on test (MSE 0.287884); the ensemble
also beat that fair calibrated comparator by 0.003519, CI
[-0.005473, -0.001590]. Validation improvement was larger (2.15% against the
calibrated public comparator), so the held-out effect is the result to trust.

The ensemble's gain over its best single member was negligible on validation:
0.000004 MSE with a confidence interval spanning zero. Use the single member
when rollout throughput matters more than the last numerical sliver.

These are value-fit results, not playing-strength results. A new PPO campaign
and fresh seat-balanced game evaluation are required before promoting any actor.

## PPO use

`train_puffer_fast.py` activates the promoted asymmetric MLP by default. Use
`--privileged-critic CHECKPOINT` to override it or `--no-privileged-critic` to
use the actor's public head. The asymmetric path automatically uses the
in-process vector collector so privileged
snapshots never cross the actor boundary, substitutes external values in GAE,
defaults the actor's obsolete public value loss coefficient to zero, updates
the external critic from the same on-policy games, and writes restartable actor
and critic checkpoints.

```bash
python train_puffer_fast.py \
  --resume models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt \
  --output-dir live_training/puffer_privileged \
  --schedule current_selfplay,ppo_champion,smart_rule,smarter_rule,historical_temporal \
  --player-count-schedule 2,3,4,5,6,7,8
```

Scalar Bayesian-tree rollouts below seven players are rejected in this mode;
their live `Game` path does not yet expose the audited vector snapshot contract.

A four-game, mixed 2/3-player GPU smoke test of the new no-flag default completed
328 decisions, used the promoted MLP for GAE, updated it, saved the critic with
the matching `0.9330329915368074` target gamma, and reloaded the result
successfully. It is only an integration test, not playing-strength evidence.

Machine-readable evidence lives in
[`benchmarks/privileged_critic`](benchmarks/privileged_critic), especially
[`mlp_gamma025at20_confirmation.json`](benchmarks/privileged_critic/mlp_gamma025at20_confirmation.json),
with the compact promotion record in
[`models/best/ppo_omniscient_mlp_gamma025at20_2to8_20260814_manifest.json`](models/best/ppo_omniscient_mlp_gamma025at20_2to8_20260814_manifest.json).
