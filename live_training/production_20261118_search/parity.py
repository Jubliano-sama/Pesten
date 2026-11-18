"""Record native public decisions and replay them in the actual serving image."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import torch

ROOT = Path.cwd() if Path("web_temporal_player.py").exists() else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CAMPAIGN = ROOT / "live_training/improvement_20261118"


@torch.inference_mode()
def record(path):
    sys.path.insert(0, str(CAMPAIGN))
    import benchmark_multiplayer_winrate as bench
    from temporal_puffer_agent import temporal_agent_factory
    from time_decisions import collect

    bench.initialize(str(CAMPAIGN / "card_belief/profile.json"), "lookahead")
    policies = bench.engine_eval._POLICIES
    games = collect(temporal_agent_factory(policies["baseline"], deterministic=True), 2, 3_447_314_801)
    episodes = []
    for opponent, players, decisions in games:
        states = {key: policy.initial_state() for key, policy in policies.items()}
        steps = []
        for decision in decisions:
            legal = [i for i, allowed in enumerate(decision.legal_actions) if allowed]
            predictions = {}
            for key, policy in policies.items():
                step = policy.step(decision, states[key])
                states[key] = step.state
                predictions[key] = step.logits[legal].tolist()
            steps.append(dict(observation=decision.observation.values, legal=legal,
                              events=[e.as_tuple() for e in decision.public_events],
                              scores=predictions["candidate"], teacher=predictions["baseline"]))
        episodes.append(dict(opponent=opponent, players=players, steps=steps))
    with gzip.open(path, "wt", encoding="utf8") as file:
        json.dump(episodes, file, separators=(",", ":"))


@torch.inference_mode()
def check(path):
    from production_temporal_policy import encode_decision_token, encode_event_token
    from temporal_observation import DecisionInput, EventKind, PlayerPrivateObservation, PublicEventToken
    from web_temporal_player import load_shared_model

    model, metadata = load_shared_model()
    with gzip.open(path, "rt", encoding="utf8") as file:
        episodes = json.load(file)
    per_size = {n: dict(decisions=0, different_from_teacher=0) for n in (2, 3, 4)}
    max_error = 0.
    for episode in episodes:
        state, history = model.initial_state(), None
        counts = per_size[episode["players"]]
        for row in episode["steps"]:
            legal = row["legal"]
            decision = DecisionInput(PlayerPrivateObservation(tuple(row["observation"])),
                                     tuple(i in legal for i in range(55)),
                                     tuple(PublicEventToken(EventKind(event[0]), *event[1:]) for event in row["events"]))
            for event in decision.public_events:
                _, _, state = model(torch.tensor([encode_event_token(event)]), state)
            logits, _, state = model(torch.tensor([encode_decision_token(decision)]), state)
            scored, history = model.public_search.score(decision, history, logits)
            for actual, key in ((logits, "teacher"), (scored, "scores")):
                expected = torch.tensor(row[key])
                got = actual[0, legal]
                assert torch.isfinite(got).all() and got.argmax() == expected.argmax()
                max_error = max(max_error, float((got - expected).abs().max()))
            counts["different_from_teacher"] += int(logits[0, legal].argmax() != scored[0, legal].argmax())
            counts["decisions"] += 1
    assert max_error < 1e-4 and all(row["different_from_teacher"] for row in per_size.values())
    print(json.dumps(dict(profile_sha256=metadata["profile_sha256"], games=len(episodes),
                          players=per_size, max_legal_logit_error=max_error,
                          differing_selected_actions=0), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (record if args.mode == "record" else check)(args.path)
