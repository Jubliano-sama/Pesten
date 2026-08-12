import pytest
import torch


pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

from puffer_distillation import (
    distill_puffer_policy,
    evaluate_puffer_distillation,
    initialize_widened_min_gru,
)
from temporal_observation import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    PASS_ACTION,
    DecisionInput,
    PlayerPrivateObservation,
)
from temporal_puffer_agent import TrajectoryStep
from temporal_puffer_v4 import NativePufferTemporalPolicy


def _strategic_decision() -> DecisionInput:
    legal = [False] * ACTION_COUNT
    legal[4] = True
    legal[PASS_ACTION] = True
    return DecisionInput(
        PlayerPrivateObservation((0.0,) * OBSERVATION_SIZE),
        tuple(legal),
    )


def test_larger_student_distills_distribution_value_and_strategic_pass():
    torch.manual_seed(31)
    teacher = NativePufferTemporalPolicy(
        hidden_size=12,
        num_layers=1,
        mode_conditioned_heads=True,
    )
    with torch.no_grad():
        teacher.puffer_policy.decoder.card_and_pass.bias[50] = 5.0
    student = NativePufferTemporalPolicy(
        hidden_size=20,
        num_layers=2,
        mode_conditioned_heads=True,
    )
    before = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
    }
    decision = _strategic_decision()
    trajectory = (
        TrajectoryStep(decision, PASS_ACTION, -0.2, 0.0),
        TrajectoryStep(decision, PASS_ACTION, -0.2, 0.0),
    )
    episodes = [(trajectory, index % 2 == 0) for index in range(8)]
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-3)

    curve = distill_puffer_policy(
        student,
        teacher,
        episodes,
        optimizer,
        epochs=2,
        minibatch_trajectories=2,
        seed=41,
        temperature=1.5,
        strategic_pass_weight=2.0,
    )

    assert len(curve) == 2
    assert curve[-1]["samples"] == 16
    assert curve[-1]["strategic_pass_opportunities"] == 16
    assert curve[-1]["teacher_strategic_pass_rate"] == 1.0
    assert 0.0 <= curve[-1]["student_strategic_pass_rate"] <= 1.0
    assert torch.isfinite(torch.tensor(curve[-1]["loss"]))
    assert any(
        not torch.equal(before[name], value)
        for name, value in student.state_dict().items()
    )


def test_function_preserving_widening_is_exact_on_batched_sequences():
    torch.manual_seed(59)
    teacher = NativePufferTemporalPolicy(
        hidden_size=12,
        num_layers=2,
        mode_conditioned_heads=True,
    )
    student = NativePufferTemporalPolicy(
        hidden_size=28,
        num_layers=2,
        mode_conditioned_heads=True,
    )
    report = initialize_widened_min_gru(student, teacher)
    tokens = torch.randn(5, 37, 100)
    teacher_logits, teacher_values = teacher.forward_native_sequence(tokens)
    student_logits, student_values = student.forward_native_sequence(tokens)

    assert report["new_channels"] == 16
    assert torch.allclose(student_logits, teacher_logits, atol=1e-6, rtol=0)
    assert torch.allclose(student_values, teacher_values, atol=1e-6, rtol=0)


def test_distillation_evaluation_reports_exact_widened_match():
    torch.manual_seed(61)
    teacher = NativePufferTemporalPolicy(
        hidden_size=12,
        num_layers=1,
        mode_conditioned_heads=True,
    )
    student = NativePufferTemporalPolicy(
        hidden_size=20,
        num_layers=1,
        mode_conditioned_heads=True,
    )
    initialize_widened_min_gru(student, teacher)
    decision = _strategic_decision()
    trajectory = (
        TrajectoryStep(decision, PASS_ACTION, -0.2, 0.0),
        TrajectoryStep(decision, PASS_ACTION, -0.2, 0.0),
    )
    metrics = evaluate_puffer_distillation(
        student, teacher, [(trajectory, True)]
    )

    assert metrics["action_agreement"] == 1.0
    assert metrics["forward_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["value_mse"] == 0.0
    assert metrics["max_legal_logit_error"] <= 1e-7
