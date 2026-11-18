import numpy as np
import torch

from opponent_style import OpponentStyle, style_features
from public_lookahead import EMPTY_WEIGHTS, HISTORY_SIZE, _batch_values


def test_style_sequence_matches_incremental_decisions_and_skips_event_updates():
    rng = np.random.default_rng(9318)
    observations = rng.uniform(0, 4, (3, 94)).astype(np.float32)
    x = torch.from_numpy(style_features(observations))
    torch.manual_seed(81)
    model = OpponentStyle(x.shape[1]).eval()
    expected, state = model(x.unsqueeze(0))
    replay = model.initial_state(1, "cpu")
    for index, obs in enumerate(observations):
        tokens = torch.zeros(1, 100)
        tokens[0, :94] = torch.from_numpy(obs)
        _, after_event = model.observe(tokens, replay)
        torch.testing.assert_close(after_event, replay)
        tokens[:, 99] = 1
        probability, replay = model.observe(tokens, after_event)
        np.testing.assert_allclose(probability[0], float(expected[0, index].sigmoid().detach()), rtol=1e-6)
    torch.testing.assert_close(replay, state)


def test_neutral_style_probability_matches_default_for_odd_and_even_budgets():
    obs = np.zeros((2, 100), dtype=np.float32)
    obs[:, [2, 6, 16, 50, 58, 89, 99]] = 1
    obs[:, [69, 70, 91, 92]] = [3, 4, 3, 2]
    states = np.zeros((2, HISTORY_SIZE), dtype=np.float32)
    for count in (7, 8):
        expected = _batch_values(obs, states, count, 12, False)
        actual = _batch_values(obs, states, count, 12, False, EMPTY_WEIGHTS, EMPTY_WEIGHTS,
                               np.full(2, .5, dtype=np.float32))
        np.testing.assert_array_equal(actual, expected)
