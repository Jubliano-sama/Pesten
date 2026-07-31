# Bayesian Tree Search, PPO, and Persistent Luna Benchmark

Date: 2026-07-31

## Outcome

The live production PPO checkpoint remains the strongest measured player.
The current hardcoded Bayesian tree-search agent won 408 of 1,000 games
against it (40.8%, 95% Wilson CI 37.8%-43.9%). A persistent Luna Max context
beat the tree agent 12-8, while a separate Luna Max context that started fresh
and then persisted across all PPO games lost to PPO 7-13.

| Match | Result from first player's perspective | 95% CI |
| --- | ---: | ---: |
| Current Bayesian tree vs live PPO | 408-592 (40.8%), 1,000 games | 37.8%-43.9% |
| Pre-Luna tree baseline vs Smart | 490-510 (49.0%), 1,000 games | 45.9%-52.1% |
| Pre-Luna tree baseline vs Random | 628-372 (62.8%), 1,000 games | 59.8%-65.7% |
| Persistent Luna vs Bayesian tree | 12-8 (60.0%), 20 games | 38.7%-78.1% |
| Fresh-start, then persistent Luna vs live PPO | 7-13 (35.0%), 20 games | 18.1%-56.7% |

The 20-game Luna intervals are wide. They are exploratory comparisons, not
precise rankings. The apparent ordering is PPO, Luna, then tree search, but the
Luna/tree difference is not statistically resolved by this sample.

## Live PPO checkpoint

The checkpoint was copied from the running `pesten-server` container on
`65.108.91.216`, not selected from a local training directory:

- container path: `/app/ppo_actor_100run3.pth`
- local path: `ppo_actor_100run3.pth`
- SHA-256: `aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186`
- policy during evaluation: deterministic legal-action-masked argmax

The current tree/PPO result is in
[`bayesian_tree_luna_heuristics_vs_live_ppo_20260731.json`](bayesian_tree_luna_heuristics_vs_live_ppo_20260731.json).
An earlier 2,000-game holdout, before the Luna-derived endgame weighting, was
838-1,162 (41.9%, CI 39.8%-44.1%). The new 40.8% result is not a measurable
improvement; the intervals overlap heavily.

## Hardcoded strategy

`hardcodedAgent.py` uses only the acting player's hand and public game state.
It never reads actual opponent card identities. Its main components are:

- a uniform hidden-deal prior over unseen physical cards;
- per-opponent Dirichlet posterior updates from publicly played suits/ranks;
- stronger suit evidence from a publicly announced Jack suit;
- evidence dilution when public hand growth implies later draws;
- deterministic posterior hand sampling;
- depth-limited paranoid minimax over every remaining player;
- explicit modeling of penalty stacking, 7/8/Ace tempo, Jack suit choice,
  Joker penalties, and the rule that a special card cannot finish.

The Luna traces reinforced four hardcoded priorities: preserve an ordinary
finisher, save disruptive cards for a one-card opponent, choose Jack/Joker
suits that maintain an ordinary-card chain, and pass rather than play when the
only remaining card is special. The evaluator now heavily penalizes giving a
one-card opponent an immediate ordinary-card finish and raises disruption
value when any opponent publicly reaches one card.

## Persistent Luna methodology

Each match used one `gpt-5.6-luna` context at max reasoning effort for all 20
games. Context was carried across games and allowed to compact naturally.
Seats alternated. Every game used a fresh, unrevealed seed.

The PPO match deliberately used a different, fresh Luna context from the tree
match. That PPO context was fresh only at match start; it remained the same
context throughout all 20 PPO games.

An early tree pilot replayed a deal after swapping seats. That would let a
persistent model remember what became the opponent's initial hand, so the
pilot was invalidated and excluded from every result above.

## Learning and behavioral findings

The tree-match Luna went 8-2 in the first half and 4-6 in the second. Continued
context therefore did not monotonically improve play and may have caused
over-adaptation or inconsistency. The PPO-match Luna went 3-7 and then 4-6;
that one-win change is too small to establish learning.

Observable tactics in the transcripts included:

- using public plays and declared suits as clues about likely opponent holdings;
- preserving a normal card as the legal finisher;
- retaining a penalty or suit-control card for an opponent at one or two cards;
- stacking a 2 during an active rank-2 penalty;
- choosing the largest useful remaining suit after a Jack or taken Joker;
- chaining tempo cards when they also reduced hand size.

The traces also caught a consistency error. PPO-match Luna correctly passed
once when its final 7 could not legally finish, but in game 17 it played a
final 7 and drew two. The hardcoded agent handles this deterministically.

## Anti-gaming audit

Both completed transcript audits pass:

| Audit | Tree match | PPO match |
| --- | ---: | ---: |
| Logged records | 461 | 420 |
| Luna decisions | 346 | 305 |
| Forced passes | 115 | 115 |
| Invalid responses | 0 | 0 |
| Hidden deal reused | No | No |
| Voluntary passes | 0 | 2 |
| Correct last-special passes | 4 | 1 |
| Legal but costly last-special plays | 0 | 1 |
| Protocol violations | 0 | 0 |

The allowlist audit proves that the game protocol exposed only public fields,
that every response was legal and from Luna's own visible hand, and that seeds
were unique. It cannot prove the contents of a model's private reasoning.

Artifacts:

- [`luna_max_vs_bayesian_tree_clean_20260731.json`](luna_max_vs_bayesian_tree_clean_20260731.json)
- [`luna_max_vs_bayesian_tree_clean_20260731.audit_report.json`](luna_max_vs_bayesian_tree_clean_20260731.audit_report.json)
- [`luna_max_vs_live_ppo_fresh_20260731.json`](luna_max_vs_live_ppo_fresh_20260731.json)
- [`luna_max_vs_live_ppo_fresh_20260731.audit_report.json`](luna_max_vs_live_ppo_fresh_20260731.audit_report.json)
