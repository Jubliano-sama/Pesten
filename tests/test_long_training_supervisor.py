import json

from long_training_supervisor import _segment_result_summary, _write_json_atomic


def test_segment_result_summary_drops_large_curve(tmp_path):
    manifest = tmp_path / "segment_result.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "warm_native_pufferlib_v4_segment",
                "curve": [{"epoch": index} for index in range(100)],
                "end_epoch": 64,
                "latest_checkpoint": "latest.pt",
            }
        ),
        encoding="utf-8",
    )

    summary = _segment_result_summary(manifest)

    assert "curve" not in summary
    assert summary["end_epoch"] == 64
    assert summary["manifest_path"] == str(manifest.resolve())
    assert len(summary["manifest_sha256"]) == 64


def test_write_json_atomic_replaces_valid_state(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text('{"old": true}\n', encoding="utf-8")

    _write_json_atomic(state_path, {"new": True})

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"new": True}
    assert not (tmp_path / ".state.json.tmp").exists()
