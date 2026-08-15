# Teacher GRU gamma sweep with gamma-matched critics

## Result

Use **gamma 0.99** as the candidate from this sweep. Neither later extension resolved an improvement: gamma 0.995 had a slightly positive point estimate, while gamma 0.9975 had a negative point estimate.

Gamma 0.9975 versus 0.99 was **-0.1978 normalized win-share percentage point**, with a paired 95% normal confidence interval of **[-0.5998, +0.2042] point** and a half-width of **0.4020 point**. This meets the requested 0.5-point uncertainty ceiling but includes zero.

Gamma 0.995 versus 0.99 was **+0.3230 normalized win-share percentage point**, with a paired 99% normal confidence interval of **[-0.0906, +0.7365] point** and a half-width of **0.4136 point**. The uncertainty target was met, but the interval includes zero; keep gamma 0.99 as the best-supported setting.

Against the independently trained gamma 0.97 reference, gamma 0.99 improved the primary normalized win-share metric by **+1.1209 percentage points**. Its paired 99.9% normal confidence interval is **[+0.7126, +1.5292] points**, with a half-width of **0.4083 point**. This meets the requested 0.5-point uncertainty ceiling and excludes zero.

Gamma 0.95 was worse than gamma 0.97 by **-1.3604 points**, 99.9% CI **[-1.7632, -0.9576]**, with a half-width of **0.4028 point**. It also meets the precision target and excludes zero.

| Contrast | Normalized win-share delta | Paired normal CI | Half-width | <= 0.5 point |
|---|---:|---:|---:|---:|
| gamma 0.95 - gamma 0.97 | -1.3604 pp | [-1.7632, -0.9576] pp | 0.4028 pp | yes |
| gamma 0.99 - gamma 0.97 | +1.1209 pp | [+0.7126, +1.5292] pp | 0.4083 pp | yes |
| gamma 0.995 - gamma 0.99 | +0.3230 pp | [-0.0906, +0.7365] pp (99%) | 0.4136 pp | yes |
| gamma 0.9975 - gamma 0.99 | -0.1978 pp | [-0.5998, +0.2042] pp (95%) | 0.4020 pp | yes |

The gamma 0.99 effect was positive against canonical random (+1.7453 points, 99.9% CI [+1.0254, +2.4652]) and the MLP champion (+1.0630, [+0.3834, +1.7427]). Its self-anchor lane was positive but not individually resolved at 99.9% (+0.5543, [-0.1669, +1.2755]). The predeclared aggregate is the primary result.

## Critic ordering and evidence

Yes: in the corrected experiment, every critic was hot-started and trained on its arm's new gamma before the first PPO actor update.

1. All compatible learned tensors were loaded from the same causal privileged critic, `models/best/ppo_omniscient_mlp_gamma025at20_2to8_20260814.pt` (SHA-256 `63fc1b610da5fa40838a96bb3157cb1aab328ac1a46ce1a3c823af484e0b9d6d`).
2. The source critic's old-gamma validation calibration was reset and a fresh AdamW optimizer was created.
3. The representation was frozen and the 66,049-parameter value head was trained against returns recomputed for gamma 0.95, 0.97, or 0.99. Epoch zero was ineligible for selection, so a selected checkpoint necessarily contains a new-gamma gradient update.
4. Per-player calibration was fit on the training split only, followed by a disjoint held-out validation readiness check.
5. Only after the gamma-matched critic passed that gate was the original teacher actor loaded for PPO. Rollout values and GAE came from that external critic; the critic then continued online updates after each actor update.

| Gamma | Selected critic epoch | Raw validation MSE, epoch 0 -> selected | Final held-out MSE after train-fit calibration | Prefit critic SHA-256 |
|---:|---:|---:|---:|---|
| 0.95 | 1 | 0.195534 -> 0.194878 | 0.194031 | `00fca2a5098d5cc87ff2136204aec2ce409729edbe9eddd32c21cc6fe957c428` |
| 0.97 | 2 | 0.271416 -> 0.254847 | 0.253500 | `1bfdcf826788f3c8651b229a6e55937f4abc815065674c060e78a3560fb80f96` |
| 0.99 | 4 | 0.489214 -> 0.399966 | 0.395771 | `842487ab7813bc71c746812ae6f46a161e8c588c394c892ea7c720df60a32edc` |
| 0.995 | 4 | 0.618321 -> 0.490802 | 0.482161 | `8b1ad78f180d0848ee8479108e261d6d450053085cbae915ef18a303f4817365` |
| 0.9975 | 4 | 0.716395 -> 0.564453 | 0.551785 | `aa9d96b2c2be39cee5d6bbb408457bd9955b2c31f38d6860b66e54a70b3dd132` |

Cross-gamma critic MSE values are not strength comparisons because gamma changes the return target's scale. The relevant readiness test is improvement from the same arm's epoch-zero hot start.

## Controlled PPO training

All five corrected arms were independent, not chained. Each began at teacher epoch 4387 from `models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt` (SHA-256 `f8f0aece2d759eadc60b3cbccca30c321c01da13322f7cae511a73c8da3a9f93`) with a fresh actor optimizer.

- 60 PPO phases and 122,880 games per arm; 614,400 corrected training games across five arms.
- Identical training seed, campaign range, opponent schedule, player-count schedule, learning rate, update count, GAE lambda, and KL target.
- All arms ended at epoch 4447; no phase stopped early.
- Maximum aggregate approximate KL was 0.002055, 0.002348, 0.001806, 0.002187, and 0.002379 for gamma 0.95, 0.97, 0.99, 0.995, and 0.9975, all below the 0.015 target.

Final actors:

| Gamma | Checkpoint | SHA-256 |
|---:|---|---|
| 0.95 | `live_training/teacher_gru_gamma_matched_critic_20260814/gamma_0p95/training/latest.pt` | `dc5d8ade279a678407d11a38bf6925322e79f461ff6857da97c865ec9f4e7cf6` |
| 0.97 | `live_training/teacher_gru_gamma_matched_critic_20260814/gamma_0p97/training/latest.pt` | `813bf93d1518dc5c6a0f0e1b7d84061a2c51f4a302f11bcc28ab1e9d9788ce86` |
| 0.99 | `live_training/teacher_gru_gamma_matched_critic_20260814/gamma_0p99/training/latest.pt` | `22373d842dfcbb219aaf2b47ceecfdb9827baaf137d2720276071572f4f21771` |
| 0.995 | `live_training/teacher_gru_gamma_matched_critic_20260814/gamma_0p995/training/latest.pt` | `1399cb6fadee1ae8f66fecbd5c13014e06375acdbd97e48f402ce6b1abb21432` |
| 0.9975 | `live_training/teacher_gru_gamma_matched_critic_20260814/gamma_0p9975/training/latest.pt` | `2ebf28b3e33bc8e22d9461da475cf2565f2e496e8553731bbfe7e61722b97729` |

No checkpoint was promoted or copied into `models/best`.

## Fixed precision evaluation

The evaluation allocation and stopping rule were locked before any corrected-model playing-strength result was opened. It used a fresh seed, 462,463 blocks across 21 opponent/player-count strata, complete seat rotations, and 1,704,410 games per actor. All three actors saw exactly the same deals and seats. Opponents were the unchanged teacher self-anchor, the MLP champion, and canonical random, weighted over 2-8 players using the training mixture.

The raw result is `live_training/teacher_gru_gamma_matched_critic_20260814/fixed_precision_evaluation.json` (SHA-256 `55c9137e1a9248cdce70204580f76e9b9f40b02909e733c9ad90fb253d9b0f1f`). The locked plan is `benchmarks/teacher_gru_gamma_precision_plan_20260814.json`; the compact checked-in protocol and results are in `benchmarks/teacher_gru_gamma_matched_critic_protocol_20260814.json`.

The gamma 0.995 extension used a separately locked 99% plan with a fresh seed, 277,479 blocks, 21 strata, and 1,022,647 complete seat-rotated games per actor. Its raw result is `live_training/teacher_gru_gamma_matched_critic_20260814/fixed_precision_gamma_0p995_vs_0p99_99ci.json` (SHA-256 `e62d7bee363e794179918d806e1e08ad3a2223d9da2fa966efb135e4bd194c25`). The plan and compact result are `benchmarks/teacher_gru_gamma_0p995_precision_plan_20260815.json` and `benchmarks/teacher_gru_gamma_0p995_protocol_20260815.json`.

The gamma 0.9975 extension used a separately locked 95% plan with a fresh seed, 166,488 blocks, 21 strata, and 613,593 complete seat-rotated games per actor. Its raw result is `live_training/teacher_gru_gamma_matched_critic_20260814/fixed_precision_gamma_0p9975_vs_0p99_95ci.json` (SHA-256 `7e5bac7ffb1571ed8412e645dfcfac6834e7c56582def1f2fdbac8db1ae6a110`). The plan and compact result are `benchmarks/teacher_gru_gamma_0p9975_precision_plan_20260815.json` and `benchmarks/teacher_gru_gamma_0p9975_protocol_20260815.json`.

## Limitations and correction record

- Confidence levels were locked per experiment: 99.9% for the original corrected sweep, 99% for gamma 0.995, and 95% for gamma 0.9975. Every interval meets the requested precision. Only the original gamma 0.95 and 0.99 contrasts against 0.97 exclude zero; neither later extension resolves a difference from gamma 0.99. The substantially more conservative distribution-free Hoeffding intervals still include zero, so the original results are not distribution-free significance claims.
- The bounded value-head critic prefit was chosen after two full-network hot-start attempts worsened gamma 0.95 validation MSE. That method was then locked and applied unchanged to all final arms before opening any corrected playing-strength result.
- The first gamma 0.995 critic prefit accidentally used the CLI's default batching seed. It was discarded before PPO and rerun with seed 20260814, matching the earlier corrected arms. The selected critic and actor recipes record the corrected checkpoint hash.
- A first proposed gamma 0.995 evaluation seed exceeded the engine's uint32 range after stratum offsets. A duplicate-model smoke test caught this before any gamma 0.995 result was opened; only the seed was replaced, while the allocation, confidence target, and fixed stopping rule stayed unchanged.
- The gamma 0.9975 PPO command was interrupted twice by interactive turn cancellation. Each interruption occurred after a saved phase boundary. The actor and online-critic optimizer states were restored, and the recorded campaign phases are contiguous from 7000 through 7059 with no gaps or repeats.
- An earlier exploratory gamma sweep used only the actor's public value head; it did **not** hot-start an external critic. Its result is retained separately in `TEACHER_GRU_GAMMA_SWEEP_20260814.md` and is not used for the corrected conclusion here.
