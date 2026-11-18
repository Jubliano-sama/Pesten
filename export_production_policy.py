#!/usr/bin/env python3
"""Export only the already-promoted teacher4387 to a portable CPU artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from production_temporal_policy import (
    MODEL_CONFIG, PORTABLE_FORMAT, TEACHER_CHECKPOINT_SHA256,
    ProductionTemporalPolicy, file_sha256, load_production_policy,
)


REPO = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPO / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"
DEFAULT_OUTPUT = REPO / "live_training/production_20260907/puffer_min_gru_192x2_plateau_teacher_epoch4387_portable.pt"


def export_production_policy(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("portable export cannot overwrite its source checkpoint")
    source_sha256 = file_sha256(source)
    if source_sha256 != TEACHER_CHECKPOINT_SHA256:
        raise ValueError("export requires the pinned promoted teacher4387 checkpoint")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    source_model = {"action_head": "mode_conditioned_55", **payload.get("model", {})}
    if source_model != MODEL_CONFIG:
        raise ValueError("source architecture differs from the promoted native teacher")
    weights = {}
    for name, value in payload["state_dict"].items():
        name = name.replace("puffer_policy.encoder.encoder.", "encoder.")
        name = name.replace("puffer_policy.network.layers.", "layers.")
        name = name.replace("puffer_policy.decoder.", "")
        if value.dtype != torch.float32:
            raise ValueError("source teacher weights must already be float32")
        weights[name] = value.detach().cpu().clone()
    check = ProductionTemporalPolicy()
    check.load_state_dict(weights, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(weights, temporary)
    temporary.replace(output)
    manifest = {
        "format": PORTABLE_FORMAT,
        "source_checkpoint": os.path.relpath(source, output.parent),
        "source_checkpoint_sha256": source_sha256,
        "weights_file": output.name,
        "weights_sha256": file_sha256(output),
        "weights_bytes": output.stat().st_size,
        "model": MODEL_CONFIG,
        "parameters": sum(value.numel() for value in weights.values()),
        "dtype": "float32", "device": "cpu",
        "token_shape": ["batch", 100], "state_shape": [2, "batch", 192],
        "inference": "native eval arithmetic, full observed public history, deterministic legal argmax; no suit averaging",
        "actor_information": "own hand, public state, public event history and legal actions only",
        "promotion": "portable re-encoding of existing promoted teacher4387; no new learned policy",
        "export_torch_version": str(torch.__version__),
        "export_source_sha256": file_sha256(Path(__file__)),
        "inference_source_sha256": file_sha256(REPO / "production_temporal_policy.py"),
    }
    manifest_path = output.with_suffix(".json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    loaded = load_production_policy(output)
    assert all(torch.equal(weights[name], value) for name, value in loaded.state_dict().items())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export_production_policy(args.source, args.output), indent=2, sort_keys=True))
