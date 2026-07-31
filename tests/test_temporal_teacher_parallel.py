import json
from pathlib import Path

from benchmark_model_zoo import DEFAULT_TEACHER
from temporal_teacher_parallel import (
    benchmark_worker_counts,
    canonical_imitation_bytes,
    collect_teacher_episodes_parallel,
    imitation_digest,
)


def test_parallel_collection_is_byte_and_semantically_deterministic():
    arguments = {
        "teacher_path": DEFAULT_TEACHER,
        "games": 4,
        "seed": 8123,
        "max_turns": 16,
        "torch_threads_per_worker": 1,
        "start_method": "spawn",
    }
    one_worker = collect_teacher_episodes_parallel(
        **arguments, workers=1
    )
    two_workers = collect_teacher_episodes_parallel(
        **arguments, workers=2
    )
    assert len(one_worker) == len(two_workers) == arguments["games"]
    assert one_worker == two_workers
    assert canonical_imitation_bytes(one_worker) == canonical_imitation_bytes(
        two_workers
    )
    assert imitation_digest(one_worker) == imitation_digest(two_workers)

    payload = canonical_imitation_bytes(two_workers)
    assert b"draw_pile" not in payload
    assert b"drawn_card" not in payload
    decoded = json.loads(payload)
    for episode in decoded:
        for decision in episode["safe_decisions"]:
            for event in decision["public_events"]:
                if event["kind"] == "draw_count":
                    assert set(event) == {
                        "actor_relative",
                        "count",
                        "kind",
                    }


def test_worker_benchmark_reports_only_aggregate_equal_results(tmp_path):
    output = benchmark_worker_counts(
        teacher_path=DEFAULT_TEACHER,
        games=3,
        seed=991,
        max_turns=10,
        worker_counts=(1, 2),
        torch_threads_per_worker=1,
        start_method="spawn",
    )
    assert len(output["runs"]) == 2
    assert output["runs"][0]["safe_sha256"] == output["runs"][1][
        "safe_sha256"
    ]
    assert all(
        run["byte_equal_to_one_worker"]
        and run["semantic_equal_to_one_worker"]
        for run in output["runs"]
    )
    encoded = json.dumps(output, sort_keys=True)
    assert "safe_decisions" not in encoded
    assert "actions" not in encoded
    assert "draw_pile" in encoded  # Explicit negative privacy declaration.
    assert not (tmp_path / "unexpected-episode-payload.json").exists()


def test_missing_teacher_fails_before_process_pool():
    missing = Path("/tmp/pesten-definitely-missing-teacher.pth")
    try:
        collect_teacher_episodes_parallel(
            teacher_path=missing,
            games=1,
            seed=1,
            workers=1,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing checkpoint unexpectedly succeeded")
