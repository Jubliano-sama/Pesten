#!/usr/bin/env python
import argparse
import copy
from tqdm import tqdm
import game
import randomAgent
import smartAgent
import math

def simulate_win_rate(params, num_games=18000):
    """
    Simulate a two-player game between a smart agent (with the given parameters)
    and a random agent over num_games. Returns the win rate of the smart agent.
    
    The 'params' dict is expected to have keys:
      stop_bonus, chain_bonus, hand_penalty, suit_common_weight, suit_penalty_weight,
      penalty_card_bonus_8, penalty_card_bonus_1, penalty_card_bonus_0, penalty_card_bonus_13, penalty_card_bonus_2.
      
    These are reassembled into the format expected by smartAgent.Agent.default_params.
    """
    smart_params = {
        'stop_bonus': params['stop_bonus'],
        'chain_bonus': params['chain_bonus'],
        'hand_penalty': params['hand_penalty'],
        'suit_common_weight': params['suit_common_weight'],
        'suit_penalty_weight': params['suit_penalty_weight'],
        'penalty_card_bonus': {
            8: params['penalty_card_bonus_8'],
            1: params['penalty_card_bonus_1'],
            0: params['penalty_card_bonus_0'],
            13: params['penalty_card_bonus_13'],
            2: params['penalty_card_bonus_2']
        },
        'softmax_temperature': 0,  # not used
        'end_penalty_bonus': 1.0
    }
    smartAgent.Agent.default_params = smart_params
    wins = 0
    for _ in range(num_games):
        # In a two-player game, the smart agent is placed first.
        agents = [smartAgent.Agent, randomAgent.Agent]
        g = game.Game(agents)
        _, winner = g.auto_simulate()
        if winner == 0:
            wins += 1
    return wins / num_games

def main():
    parser = argparse.ArgumentParser(
        description='Optimize smart agent parameters via adaptive gradient ascent.'
    )
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of optimization epochs (default: 1000)')
    parser.add_argument('--games', type=int, default=6000,
                        help='Number of games per evaluation (default: 3000)')
    args = parser.parse_args()

    # Initial parameter values (flattened):
    params = {
        'stop_bonus': -1.9410784240163563,
        'chain_bonus': 1.3063677649582381,
        'hand_penalty': 1.0233570225230266,
        'suit_common_weight': 0.4294193772621973,
        'suit_penalty_weight': -1.8509912660600005,
        'penalty_card_bonus_8': -1.5296405010776144,
        'penalty_card_bonus_1': 0.03399216633431756,
        'penalty_card_bonus_0': -0.03538887807151039,
        'penalty_card_bonus_13': -3.016327711854124,
        'penalty_card_bonus_2': 0.0651559971691161,
    }
    # Base learning rates for each parameter:
    base_lr = {
        'stop_bonus': 0.0015,
        'chain_bonus': 0.0015,
        'hand_penalty': 0.0015,
        'suit_common_weight': 0.0015,
        'suit_penalty_weight': 0.0015,
        'penalty_card_bonus_8': 0.0015,
        'penalty_card_bonus_1': 0.0015,
        'penalty_card_bonus_0': 0.0015,
        'penalty_card_bonus_13': 0.0015,
        'penalty_card_bonus_2': 0.0015,
    }
    # We'll update lr adaptively.
    lr = copy.deepcopy(base_lr)

    # For adaptive learning rates, maintain an exponential moving average of squared gradients.
    avg_sq_grad = {key: 0.0 for key in params}
    decay = 0.9
    epsilon = 1e-8

    best_params = copy.deepcopy(params)
    best_win_rate = simulate_win_rate(params, args.games)
    print("Initial parameters:", params, "Win rate:", best_win_rate)

    for epoch in tqdm(range(args.epochs), desc="Optimizing", unit="epoch"):
        current_win_rate = simulate_win_rate(params, args.games)
        gradients = {}
        # Compute finite-difference gradient for each parameter using a relative delta of 10%.
        for key in params:
            # Compute delta as 10% of the absolute value (or use 0.01 if too small)
            delta_val = 0.2 * abs(params[key]) if abs(params[key]) > 1e-6 else 0.01
            params_perturbed = copy.deepcopy(params)
            params_perturbed[key] += delta_val
            win_rate_perturbed = simulate_win_rate(params_perturbed, args.games)
            grad = (win_rate_perturbed - current_win_rate) / delta_val
            gradients[key] = grad

        # Update adaptive learning rates.
        for key in params:
            avg_sq_grad[key] = decay * avg_sq_grad[key] + (1 - decay) * (gradients[key] ** 2)
            lr[key] = base_lr[key] / (math.sqrt(avg_sq_grad[key]) + epsilon)

        # Gradient ascent update (capped to the delta for each parameter).
        for key in params:
            current_val = params[key]
            # Recompute delta for this parameter.
            delta_val = 0.2 * abs(current_val) if abs(current_val) > 1e-6 else 0.01
            update_value = lr[key] * gradients[key]
            # Cap the update's magnitude to delta_val.
            if abs(update_value) > delta_val:
                update_value = math.copysign(delta_val, update_value)
            params[key] += update_value

        new_win_rate = simulate_win_rate(params, args.games)
        if new_win_rate > best_win_rate:
            best_win_rate = new_win_rate
            best_params = copy.deepcopy(params)
        tqdm.write(f"Epoch {epoch+1}: params: {params}, win rate: {new_win_rate:.3f}")

    print("Optimization complete.")
    print("Best parameters:", best_params, "Best win rate:", best_win_rate)

if __name__ == '__main__':
    main()
