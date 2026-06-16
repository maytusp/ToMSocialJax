"""Perspective transforms for egocentric SocialJax observations."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class TransformMetadata(NamedTuple):
    self_channel: int
    other_channel: int
    angle_start: int
    angle_count: int


DEFAULT_METADATA = TransformMetadata(
    self_channel=4,
    other_channel=5,
    angle_start=6,
    angle_count=4,
)

TASK_METADATA = {
    "clean_up": DEFAULT_METADATA,
    "clean_up_single": DEFAULT_METADATA,
    "coin_game": DEFAULT_METADATA,
    "coin_game_single": DEFAULT_METADATA,
    "gift": DEFAULT_METADATA,
    "harvest_common_open": DEFAULT_METADATA,
    "harvest_common_single": DEFAULT_METADATA,
    "mushrooms": DEFAULT_METADATA,
    "pd_arena": DEFAULT_METADATA,
    "territory_open": DEFAULT_METADATA,
    "coop_mining": TransformMetadata(
        self_channel=6,
        other_channel=7,
        angle_start=8,
        angle_count=4,
    ),
}


def resolve_transform_metadata(
    task: str = "coin_game",
    *,
    self_channel: int | None = None,
    other_channel: int | None = None,
    angle_start: int | None = None,
    angle_count: int | None = None,
) -> TransformMetadata:
    """Resolve task-specific observation channels, with explicit overrides."""
    metadata = TASK_METADATA.get(task, DEFAULT_METADATA)
    return TransformMetadata(
        self_channel=metadata.self_channel if self_channel is None else self_channel,
        other_channel=metadata.other_channel if other_channel is None else other_channel,
        angle_start=metadata.angle_start if angle_start is None else angle_start,
        angle_count=metadata.angle_count if angle_count is None else angle_count,
    )


def _first_channel_position(
    obs: jnp.ndarray,
    channel: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mask = obs[..., channel] > 0
    flat_idx = jnp.argmax(mask.reshape(-1))
    width = obs.shape[1]
    row = flat_idx // width
    col = flat_idx % width
    return row, col, jnp.any(mask)


def _rotate_offsets(
    row_offsets: jnp.ndarray,
    col_offsets: jnp.ndarray,
    k: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate row/column offsets using the same k convention as jnp.rot90."""
    rot_row = jnp.where(k == 1, -col_offsets, row_offsets)
    rot_col = jnp.where(k == 1, row_offsets, col_offsets)

    rot_row = jnp.where(k == 2, -row_offsets, rot_row)
    rot_col = jnp.where(k == 2, -col_offsets, rot_col)

    rot_row = jnp.where(k == 3, col_offsets, rot_row)
    rot_col = jnp.where(k == 3, -row_offsets, rot_col)
    return rot_row, rot_col


@partial(
    jax.jit,
    static_argnames=(
        "task",
        "self_channel",
        "other_channel",
        "angle_start",
        "angle_count",
    ),
)
def transform_ego_to_partner_obs(
    ego_obs: jnp.ndarray,
    *,
    task: str = "coin_game",
    self_channel: int | None = None,
    other_channel: int | None = None,
    angle_start: int | None = None,
    angle_count: int | None = None,
) -> jnp.ndarray:
    """Transform one ego observation into the visible partner's frame.

    The transform uses only information already present in ``ego_obs``. If the
    partner is not visible in ``other_channel``, the function returns zeros with
    the same shape and dtype.
    """
    metadata = resolve_transform_metadata(
        task,
        self_channel=self_channel,
        other_channel=other_channel,
        angle_start=angle_start,
        angle_count=angle_count,
    )
    self_channel = metadata.self_channel
    other_channel = metadata.other_channel
    angle_start = metadata.angle_start
    angle_count = metadata.angle_count

    height, width, _ = ego_obs.shape
    self_row, self_col, has_self = _first_channel_position(ego_obs, self_channel)
    partner_row, partner_col, has_partner = _first_channel_position(
        ego_obs,
        other_channel,
    )

    partner_angle = ego_obs[
        partner_row,
        partner_col,
        angle_start : angle_start + angle_count,
    ]
    partner_dir = jnp.argmax(partner_angle)

    rows, cols = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.int32),
        jnp.arange(width, dtype=jnp.int32),
        indexing="ij",
    )
    row_offsets = rows - partner_row
    col_offsets = cols - partner_col
    rot_row_offsets, rot_col_offsets = _rotate_offsets(
        row_offsets,
        col_offsets,
        partner_dir,
    )
    target_rows = self_row + rot_row_offsets
    target_cols = self_col + rot_col_offsets

    valid = (
        (target_rows >= 0)
        & (target_rows < height)
        & (target_cols >= 0)
        & (target_cols < width)
        & has_self
        & has_partner
    )
    clipped_rows = jnp.clip(target_rows, 0, height - 1)
    clipped_cols = jnp.clip(target_cols, 0, width - 1)
    moved = jnp.zeros_like(ego_obs).at[clipped_rows, clipped_cols, :].add(
        ego_obs * valid[..., None].astype(ego_obs.dtype)
    )

    transformed = moved
    transformed = transformed.at[..., self_channel].set(moved[..., other_channel])
    transformed = transformed.at[..., other_channel].set(moved[..., self_channel])

    angle_slice = slice(angle_start, angle_start + angle_count)
    transformed = transformed.at[..., angle_slice].set(0)
    ego_relative_dir = (-partner_dir) % angle_count
    ego_angle = jax.nn.one_hot(ego_relative_dir, angle_count, dtype=ego_obs.dtype)
    visible_old_ego = transformed[..., other_channel] > 0
    transformed = transformed.at[..., angle_slice].set(
        ego_angle * visible_old_ego[..., None].astype(ego_obs.dtype)
    )

    return jnp.where(has_partner, transformed, jnp.zeros_like(ego_obs))


@partial(
    jax.jit,
    static_argnames=(
        "task",
        "self_channel",
        "other_channel",
        "angle_start",
        "angle_count",
    ),
)
def batch_transform_ego_to_partner_obs(
    obs: jnp.ndarray,
    *,
    task: str = "coin_game",
    self_channel: int | None = None,
    other_channel: int | None = None,
    angle_start: int | None = None,
    angle_count: int | None = None,
) -> jnp.ndarray:
    """Apply ``transform_ego_to_partner_obs`` over any leading batch dimensions."""
    metadata = resolve_transform_metadata(
        task,
        self_channel=self_channel,
        other_channel=other_channel,
        angle_start=angle_start,
        angle_count=angle_count,
    )
    obs_shape = obs.shape
    flat_obs = obs.reshape((-1,) + obs_shape[-3:])
    transformed = jax.vmap(
        partial(
            transform_ego_to_partner_obs,
            task=task,
            self_channel=metadata.self_channel,
            other_channel=metadata.other_channel,
            angle_start=metadata.angle_start,
            angle_count=metadata.angle_count,
        )
    )(flat_obs)
    return transformed.reshape(obs_shape)
