# Native PufferLib 4 runtime

This isolated uv project pins the official PufferAI/PufferLib `4.0` branch at
commit `c5d3c637446047a6efbcaa74c039c5295d201ab0` and uses uv-managed CPython
3.12.12. It does not modify the repository's legacy `venv`.

Install the Python API:

```bash
UV_CACHE_DIR=/tmp/pesten-puffer4-uv-cache uv python install 3.12
UV_CACHE_DIR=/tmp/pesten-puffer4-uv-cache uv sync \
  --project puffer4-native --python 3.12
```

Verify the pinned API:

```bash
puffer4-native/.venv/bin/python -c \
  'import pufferlib, pufferlib.models; print(pufferlib.__version__, pufferlib.models.MinGRU)'
```

The v4 native `_C` extension is built per environment by the upstream
`build.sh`; installing the Python package alone intentionally does not fabricate
that backend. The reproducible upstream-native smoke procedure is:

```bash
git clone --branch 4.0 --single-branch https://github.com/PufferAI/PufferLib.git /tmp/pufferlib4-src
git -C /tmp/pufferlib4-src checkout c5d3c637446047a6efbcaa74c039c5295d201ab0
cd /tmp/pufferlib4-src
PATH=/home/tommie/Pesten/puffer4-native/.venv/bin:$PATH \
  ./build.sh squared --cpu
```

At the pinned revision the upstream `benchmark` binding does not compile
(`OBS_TENSOR_T` and `rng` are stale in that target); `squared` is the supported
native CPU smoke used here. Pesten's temporal integration itself exercises the
real v4 `Policy`, `MinGRU`, parallel sequence replay, and native `_C` trainer
through separate tests and benchmark commands:

```bash
puffer4-native/.venv/bin/python -m pytest -q \
  tests/test_temporal_privacy.py \
  tests/test_temporal_smoke.py \
  tests/test_temporal_puffer_v4.py \
  tests/test_temporal_batched_speed.py

puffer4-native/.venv/bin/python benchmark_temporal_batched_speed.py \
  --device cuda --games 50

puffer4-native/.venv/bin/python temporal_reward_league.py \
  --device cuda
```

Pesten itself remains a Python environment, so its adapter uses the pinned
native v4 `Policy` and `MinGRU` API while keeping the immutable privacy schema
in `temporal_observation.py`. Actor and critic receive only the current
player's immutable private observation, legal-action mask, ordered public event
tokens, and that player's recurrent state. Opponent hands, drawn-card
identities, draw-pile contents, and raw `Game`/`BatchedPesten` objects never
cross the policy boundary.
