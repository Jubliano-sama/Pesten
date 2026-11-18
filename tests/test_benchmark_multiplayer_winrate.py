import math
import json
import time

import pytest

import benchmark_multiplayer_winrate as benchmark
import benchmark_rlvr_pool_batched as engine_eval


def test_freeze_rejects_changed_asset_and_different_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"first")
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({
        "candidate": model.name, "candidate_kind": "native",
        "files": {model.name: benchmark.file_sha256(model)},
    }))
    assert benchmark.verify_freeze(manifest, model, "native") == benchmark.file_sha256(manifest)
    with pytest.raises(ValueError, match="requested candidate"):
        benchmark.verify_freeze(manifest, model, "residual")
    model.write_bytes(b"second")
    with pytest.raises(ValueError, match="file mismatch"):
        benchmark.verify_freeze(manifest, model, "native")


def test_progress_discloses_completion_only_and_keeps_rows(capsys):
    rows = [{"wins": 7}] * 20
    assert benchmark.collect_with_progress(rows, len(rows), time.monotonic()) == rows
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10
    for index, line in enumerate(lines, 1):
        assert f"({10 * index}%)" in line and "elapsed" in line and "wins" not in line


def test_resume_reuses_exact_chunks_through_torn_tail_and_rejects_mismatch(tmp_path):
    from multiplayer_eval_checkpoints import chunk_log
    path = tmp_path / "chunks.jsonl"
    tasks = benchmark.make_tasks(434200117, 2, 1)
    header = {"freeze_manifest_sha256": "immutable", "chunk_blocks": 1}
    task = tasks[0]
    row = {"seed": task["seed"], "players": 2, "opponent": task["opponent"], "totals": {}}
    for name in benchmark.VARIANTS:
        for outcome, value in (("win", .5), ("loss", .5), ("draw", 0.)):
            row[f"{name}_{outcome}_shares"] = [value]
        row[f"{name}_scores"] = [.5]
        row["totals"][name] = {"wins": 1, "losses": 1, "turns": 40}
    with chunk_log(path, header=header, tasks=tasks, resume=False) as (saved, append):
        assert not saved
        append(row)
    committed = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"unfinished"')
    with chunk_log(path, header=header, tasks=tasks, resume=True) as (saved, append):
        assert list(saved.values()) == [row] and path.read_bytes() == committed
        with pytest.raises(ValueError, match="duplicate"):
            append(row)
        with pytest.raises(ValueError, match="totals"):
            append({**row, "seed": tasks[1]["seed"], "candidate_win_shares": [0.]})
    assert [p.read_bytes() for p in tmp_path.glob("*.incomplete-*")] == [b'{"unfinished"']
    with pytest.raises(ValueError, match="different freeze"):
        with chunk_log(path, header={**header, "freeze_manifest_sha256":"other"}, tasks=tasks, resume=True):
            pass
    with pytest.raises(ValueError, match="task plan"):
        with chunk_log(path, header=header, tasks=tasks[:-1], resume=True):
            pass
    path.write_bytes(committed.replace(b'"turns":40', b'"turns":41'))
    with pytest.raises(ValueError, match="checksum"):
        with chunk_log(path, header=header, tasks=tasks, resume=True):
            pass


@pytest.mark.parametrize("path,total", [
    ("protocol.json", 57600), ("selection/standalone_protocol.json", 28800),
])
def test_freezer_normalizes_original_protocols_and_rejects_incompatible_layout(path, total):
    from live_training.improvement_20261118.freeze_candidate import holdout_allocation
    directory = benchmark.ROOT / "live_training/improvement_20261118"
    data = json.loads((directory / path).read_bytes())
    result = holdout_allocation(data)
    bank = data.get("holdout", data)
    assert result["base_seed"] == bank["base_seed"] and result["total_games"] == total
    assert result["blocks_per_cell"] == total // 36
    bank["total_games"] -= 1
    with pytest.raises(ValueError, match="allocation"):
        holdout_allocation(data)


def test_cell_banks_are_disjoint_and_focal_seat_rotates_against_copies():
    tasks = benchmark.make_tasks(127, blocks=3, chunk_blocks=2)
    banks = set()
    for task in tasks:
        blocks = set(range(task["seed"], task["seed"] + task["blocks"]))
        assert banks.isdisjoint(blocks)
        banks.update(blocks)
        records, roles, seeds = engine_eval._make_layout(task)
        assert {int(seed) for seed in seeds} == blocks
        players = task["players"]
        assert len(records) == task["blocks"] * 2 * players
        for (block, variant, focal, seed), table in zip(records, roles):
            assert block == seed - task["seed"]
            opponent = ("temporal", "baseline") if task["opponent"] == "self_anchor" else ("canonical_random", None)
            assert table == [("temporal", variant) if seat == focal else opponent for seat in range(players)]
    assert len(banks) == 6 * 3


def test_draws_and_weights_and_block_uncertainty():
    rows = []
    for task in benchmark.make_tasks(17, blocks=2, chunk_blocks=1):
        row = {**task, "totals": {}}
        players = task["players"]
        for name in benchmark.VARIANTS:
            # First cell and first deal only: all focal seats win under the candidate.
            won = int(name == "candidate" and task["seed"] == 17)
            row[f"{name}_win_shares"] = [won]
            row[f"{name}_draw_shares"] = [1 - won]
            row[f"{name}_loss_shares"] = [0]
            row["totals"][name] = {"wins": won * players, "draws": (1 - won) * players, "turns": 32}
        rows.append(row)
    report = benchmark.summarize(rows[::-1], seed=17, blocks=2)
    assert report["raw_blocks"][0]["paired_win_rate_deltas"] == [1., 0.]
    cell = report["cells"][0]
    assert cell["candidate"]["win_rate"] == .5
    assert cell["baseline"]["win_rate"] == 0
    assert cell["standard_error"] == pytest.approx(.5)
    assert report["macro"]["delta"] == pytest.approx(.5 / 6)
    assert report["macro"]["standard_error"] == pytest.approx(.5 / 6)
    assert report["by_opponent"]["self_anchor"]["delta"] == pytest.approx(.5 / 3)
    assert report["by_players"][2]["delta"] == pytest.approx(.5 / 2)
    lower, upper = report["macro"]["paired_ci95"]
    assert lower < 0 < upper and math.isclose(upper - lower, 2 * benchmark.Z95 * .5 / 6)
    with pytest.raises(ValueError, match="missing or duplicated"):
        benchmark.summarize([*rows, rows[0]], seed=17, blocks=2)


@pytest.fixture(scope="module")
def teacher_factory():
    puffer = pytest.importorskip("pufferlib")
    if not str(getattr(puffer, "__version__", "")).startswith("4"):
        pytest.skip("native temporal teacher requires PufferLib 4")
    from temporal_puffer_agent import temporal_agent_factory
    benchmark.initialize(benchmark.TEACHER, "native")
    return temporal_agent_factory(engine_eval._POLICIES["baseline"], deterministic=True)


@pytest.mark.parametrize("opponent,players", benchmark.CELLS)
def test_batched_results_match_scalar_game_and_actual_randomagent(teacher_factory, opponent, players):
    from benchmark_rlvr_pool import _candidate_score
    from randomAgent import AgentRandom

    seed = 418720000 + players
    factories = teacher_factory if opponent == "self_anchor" else AgentRandom
    _, wins, losses, draws, turns = _candidate_score(
        teacher_factory, factories, players=players, seed=seed,
    )
    row = benchmark.evaluate_task({
        "opponent": opponent, "players": players, "seed": seed, "blocks": 1,
        "variants": benchmark.VARIANTS, "policy_names": benchmark.VARIANTS, "self_anchor_name": "baseline",
    })
    for variant in benchmark.VARIANTS:
        totals = row["totals"][variant]
        assert tuple(totals.get(key, 0) for key in ("wins", "losses", "draws", "turns")) == (wins, losses, draws, turns)
