"""Print a real CoinGame obs before and after partner-perspective transform.

Run from the repo root:

    python tests/sanity_check_perspective_transform.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import socialjax  # noqa: E402
from algorithms.perspective_transform import transform_ego_to_partner_obs  # noqa: E402


SELF_CHANNEL = 4
OTHER_CHANNEL = 5
ANGLE_START = 6
ANGLE_COUNT = 4

CLOSE_SPAWN_MAP = [
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCPCCCCC",
    "CCCCCPCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
    "CCCCCCCCCCC",
]


def _channel_grid(obs, channel):
    return jnp.asarray(obs[..., channel], dtype=jnp.int8)


def _print_obs(label, obs):
    print(f"\n{label} full tensor:")
    print(obs)
    print(f"\n{label} self channel:")
    print(_channel_grid(obs, SELF_CHANNEL))
    print(f"\n{label} other channel:")
    print(_channel_grid(obs, OTHER_CHANNEL))
    print(f"\n{label} relative-orientation channels:")
    for idx in range(ANGLE_COUNT):
        print(f"angle {idx}:")
        print(_channel_grid(obs, ANGLE_START + idx))


def main():
    env = socialjax.make(
        "coin_game",
        jit=False,
        map_ASCII=CLOSE_SPAWN_MAP,
        grid_size=(16, 11),
        obs_size=3,
    )

    obs, state = env.reset(jax.random.PRNGKey(0))
    ego_agent = 0
    ego_obs = obs[ego_agent]
    transformed_obs = transform_ego_to_partner_obs(ego_obs)

    print("agent_locs [row, col, direction]:")
    print(state.agent_locs)
    print(f"\nego agent: {ego_agent}")
    print(f"partner visible in original obs: {bool(jnp.any(ego_obs[..., OTHER_CHANNEL]))}")
    print(f"transform returned all zeros: {bool(jnp.all(transformed_obs == 0))}")

    _print_obs("original ego_obs", ego_obs)
    _print_obs("transformed partner_obs", transformed_obs)


if __name__ == "__main__":
    main()
