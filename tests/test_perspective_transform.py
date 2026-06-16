import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.perspective_transform import (  # noqa: E402
    batch_transform_ego_to_partner_obs,
    resolve_transform_metadata,
    transform_ego_to_partner_obs,
)


def _base_obs(partner_dir=0, partner_pos=(1, 2), self_pos=(2, 2)):
    obs = jnp.zeros((5, 5, 14), dtype=jnp.int8)
    obs = obs.at[self_pos[0], self_pos[1], 4].set(1)
    obs = obs.at[partner_pos[0], partner_pos[1], 5].set(1)
    obs = obs.at[partner_pos[0], partner_pos[1], 6 + partner_dir].set(1)
    return obs


def test_partner_visible_straight_ahead_becomes_self_at_canonical_position():
    obs = _base_obs(partner_dir=0)

    transformed = transform_ego_to_partner_obs(obs)

    assert transformed[2, 2, 4] == 1
    assert transformed[2, 2, 5] == 0
    assert transformed[3, 2, 5] == 1
    assert transformed[3, 2, 6] == 1


def test_partner_relative_orientation_rotates_visible_cells_and_updates_ego_angle():
    item_positions = [(1, 2), (2, 1), (3, 2), (2, 3)]
    ego_positions = [(3, 2), (2, 3), (1, 2), (2, 1)]
    ego_angle_channels = [6, 9, 8, 7]

    for partner_dir in range(4):
        obs = _base_obs(partner_dir=partner_dir)
        obs = obs.at[0, 2, 0].set(1)

        transformed = transform_ego_to_partner_obs(obs)
        item_row, item_col = item_positions[partner_dir]
        ego_row, ego_col = ego_positions[partner_dir]

        assert transformed[item_row, item_col, 0] == 1
        assert transformed[ego_row, ego_col, 5] == 1
        assert transformed[ego_row, ego_col, ego_angle_channels[partner_dir]] == 1


def test_shifted_out_cells_are_zero_filled_not_wrapped():
    obs = _base_obs(partner_dir=0, partner_pos=(0, 0), self_pos=(2, 2))
    obs = obs.at[4, 4, 0].set(1)

    transformed = transform_ego_to_partner_obs(obs)

    assert transformed[2, 2, 4] == 1
    assert transformed[..., 0].sum() == 0


def test_missing_partner_returns_zero_observation():
    obs = jnp.zeros((5, 5, 14), dtype=jnp.int8)
    obs = obs.at[2, 2, 4].set(1)

    transformed = transform_ego_to_partner_obs(obs)

    assert jnp.array_equal(transformed, jnp.zeros_like(obs))


def test_batch_transform_matches_single_transform_and_jit():
    obs = _base_obs(partner_dir=1)
    missing_partner = jnp.zeros((5, 5, 14), dtype=jnp.int8).at[2, 2, 4].set(1)
    batched = jnp.stack([obs, missing_partner])

    transformed = batch_transform_ego_to_partner_obs(batched)

    assert jnp.array_equal(transformed[0], transform_ego_to_partner_obs(obs))
    assert jnp.array_equal(
        transformed[1],
        transform_ego_to_partner_obs(missing_partner),
    )
    assert jnp.array_equal(
        jax.jit(transform_ego_to_partner_obs)(obs),
        transform_ego_to_partner_obs(obs),
    )


def test_resolve_task_metadata_for_coop_mining():
    metadata = resolve_transform_metadata("coop_mining")

    assert metadata.self_channel == 6
    assert metadata.other_channel == 7
    assert metadata.angle_start == 8
    assert metadata.angle_count == 4


def test_explicit_channel_overrides_take_precedence():
    metadata = resolve_transform_metadata(
        "coin_game",
        self_channel=10,
        other_channel=11,
        angle_start=12,
        angle_count=3,
    )

    assert metadata.self_channel == 10
    assert metadata.other_channel == 11
    assert metadata.angle_start == 12
    assert metadata.angle_count == 3


def test_coop_mining_task_uses_task_channels():
    obs = jnp.zeros((5, 5, 12), dtype=jnp.int8)
    obs = obs.at[2, 2, 6].set(1)
    obs = obs.at[1, 2, 7].set(1)
    obs = obs.at[1, 2, 8].set(1)

    transformed = transform_ego_to_partner_obs(obs, task="coop_mining")

    assert transformed[2, 2, 6] == 1
    assert transformed[3, 2, 7] == 1
    assert transformed[3, 2, 8] == 1
