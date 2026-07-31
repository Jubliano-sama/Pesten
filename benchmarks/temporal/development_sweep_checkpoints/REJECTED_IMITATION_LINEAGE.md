# Rejected PPO-imitation lineage

These checkpoints were produced during an interrupted development sweep before
the accepted training objective changed. They may contain PPO-derived
imitation updates and must never be selected, resumed, deployed, or evaluated
as the accepted temporal agent.

The accepted lineage is random-initialized and reward-only. It is written to a
separate directory by `temporal_reward_league.py`.
