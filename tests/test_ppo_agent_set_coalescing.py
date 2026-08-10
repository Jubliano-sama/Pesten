from ppo2 import (
    coalesce_duplicate_agent_sets,
    group_compatible_agent_sets,
)


def test_duplicate_agent_sets_coalesce_in_first_seen_order() -> None:
    assert coalesce_duplicate_agent_sets(
        [
            ["smart", "rl"],
            ["SMART", "RL"],
            ["random", "rl"],
            ["self", "self", "rl"],
            ["self", "self", "rl"],
        ],
        [46, 46, 46, 46, 52],
    ) == [
        (["smart", "rl"], 92),
        (["random", "rl"], 46),
        (["self", "self", "rl"], 98),
    ]


def test_coalescing_preserves_total_episode_weight() -> None:
    scheduled = coalesce_duplicate_agent_sets(
        [["smart", "rl"], ["smart", "rl"], ["random", "rl"]],
        [3, 4, 5],
    )
    assert sum(count for _, count in scheduled) == 12


def test_compatible_sets_group_only_when_consecutive() -> None:
    scheduled = [
        (["smart", "rl"], 10),
        (["random", "rl"], 20),
        (["self", "self", "rl"], 30),
        (["smart", "smart", "rl"], 40),
        (["smarter", "rl"], 50),
    ]
    assert group_compatible_agent_sets(scheduled) == [
        [
            (["smart", "rl"], 10),
            (["random", "rl"], 20),
        ],
        [
            (["self", "self", "rl"], 30),
            (["smart", "smart", "rl"], 40),
        ],
        [
            (["smarter", "rl"], 50),
        ],
    ]
