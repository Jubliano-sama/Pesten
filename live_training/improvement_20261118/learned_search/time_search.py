"""Measure complete match replay time for one focal policy at a time on fixed deals."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import benchmark_rlvr_pool_batched as evaluator
from benchmark_multiplayer_winrate import make_tasks, orbit24
from public_lookahead import load_candidate
from strongest_policy_search import TEACHER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('profiles', nargs='+', type=Path)
    args = parser.parse_args()
    if args.blocks < 2 or args.repeats < 1:
        raise ValueError('timing needs two or more blocks and a positive repeat count')
    torch.set_num_threads(1)
    evaluator._initialize_multi_worker({'base': str(TEACHER)}, 'ppo_actor_100run3.pth')
    policies = evaluator._POLICIES
    policies['base'] = orbit24(policies['base'])
    for profile in args.profiles:
        policies[profile.stem] = load_candidate(profile)
    arms = list(policies)
    tasks = make_tasks(args.seed, args.blocks, args.blocks)
    rows = []
    def replay(label, task):
        work = {**task, 'variants': [label], 'policy_names': list(policies), 'self_anchor_name': 'base'}
        start = time.monotonic()
        result = evaluator._evaluate_task(work)
        return time.monotonic()-start, result['totals'][label]
    for label in arms:
        for task in tasks:
            replay(label, {**task, 'blocks': 2})
    for repeat in range(args.repeats):
        for i, task in enumerate(tasks):
            for label in (arms if (repeat+i)%2 == 0 else reversed(arms)):
                elapsed, totals = replay(label, task)
                row = {'repeat': repeat, 'profile': label, 'players': task['players'],
                       'opponent': task['opponent'], 'seconds': elapsed,
                       'games': task['blocks']*task['players'], 'turns': totals['turns']}
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {'scope': 'Full warmed single-thread CPU replay, real engine and every player, six matched cells, per-policy batches',
               'seed': args.seed, 'blocks': args.blocks, 'repeats': args.repeats, 'rows': rows}
    total = {}
    for label in arms:
        selected = [row for row in rows if row['profile'] == label]
        seconds = sum(row['seconds'] for row in selected)
        total[label] = {'seconds': seconds, 'games_per_second': sum(row['games'] for row in selected)/seconds,
                        'turns_per_second': sum(row['turns'] for row in selected)/seconds}
    summary['total'] = total
    args.output.write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
