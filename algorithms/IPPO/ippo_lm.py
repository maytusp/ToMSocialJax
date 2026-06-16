"""Task-agnostic two-stream recurrent IPPO with pretrained SocialJax encoders."""

import sys
from pathlib import Path
from typing import Dict, Sequence

import hydra

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import socialjax
import wandb
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from algorithms.perspective_transform import batch_transform_ego_to_partner_obs
try:
    from .ippo_rnn import (
        CNN,
        ScannedRNN,
        Transition,
        _agent_value,
        batchify_done,
        batchify_obs,
        batchify_reward,
        load_params,
        save_params,
        unbatchify,
    )
except ImportError:
    from ippo_rnn import (
        CNN,
        ScannedRNN,
        Transition,
        _agent_value,
        batchify_done,
        batchify_obs,
        batchify_reward,
        load_params,
        save_params,
        unbatchify,
    )
from socialjax.wrappers.baselines import LogWrapper


def repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)


def _strip_single_suffix(task: str) -> str:
    return task[:-7] if task.endswith("_single") else task


def resolve_transform_task(config):
    return config.get("TRANSFORM_TASK") or _strip_single_suffix(config["ENV_NAME"])


def experiment_name(config, run_name="ippo_lm"):
    configured_name = config.get("RUN_NAME")
    if configured_name:
        return configured_name
    transform_suffix = "cpt" if config.get("PERSPECTIVE_TRANSFORM", True) else "sameinp"
    self_pred_suffix = "selfpred" if config.get("USE_SELF_PRED", True) else "nopred"
    return f"{run_name}_{transform_suffix}_{self_pred_suffix}_{config['ENV_NAME']}"


class TwoStreamActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        assert obs.ndim == 5, f"Expected obs (T,B,H,W,C), got {obs.shape}"

        if self.config.get("PERSPECTIVE_TRANSFORM", True):
            other_obs = batch_transform_ego_to_partner_obs(
                obs,
                task=resolve_transform_task(self.config),
                self_channel=self.config.get("TRANSFORM_SELF_CHANNEL"),
                other_channel=self.config.get("TRANSFORM_OTHER_CHANNEL"),
                angle_start=self.config.get("TRANSFORM_ANGLE_START"),
                angle_count=self.config.get("TRANSFORM_ANGLE_COUNT"),
            )
        else:
            other_obs = obs

        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)
        flat_other_obs = other_obs.reshape(-1, h, w, c)

        self_embedding = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="self_cnn",
        )(flat_obs)
        other_embedding = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="other_cnn",
        )(flat_other_obs)

        self_embedding = self_embedding.reshape(*obs.shape[:-3], -1)
        other_embedding = other_embedding.reshape(*obs.shape[:-3], -1)

        self_embedding = nn.LayerNorm(name="self_ln")(self_embedding)
        other_embedding = nn.LayerNorm(name="other_ln")(other_embedding)

        if not self.config.get("FINETUNE_SELF_STREAM", True):
            self_embedding = jax.lax.stop_gradient(self_embedding)
        if not self.config.get("FINETUNE_OTHER_STREAM", False):
            other_embedding = jax.lax.stop_gradient(other_embedding)

        rnn_input = jnp.concatenate([self_embedding, other_embedding], axis=-1)
        hidden, embedding = ScannedRNN(name="fusion_rnn")(hidden, (rnn_input, dones))
        z = embedding

        aux = {"rnn_hidden": z}
        if self.config.get("USE_SELF_PRED", True):
            pred_z = nn.Dense(
                self.config["FC_DIM_SIZE"],
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="self_pred_fc",
            )(z)
            pred_z = activation(pred_z)
            pred_gammas = tuple(self.config.get("SELF_PRED_GAMMAS", (0.0, 0.5, 0.9)))
            pred_z = nn.Dense(
                len(pred_gammas) * z.shape[-1],
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="self_pred_out",
            )(pred_z)
            pred_z = pred_z.reshape(*z.shape[:-1], len(pred_gammas), z.shape[-1])
            aux["pred_hidden_repr"] = pred_z

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_fc",
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_fc",
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1), aux


def _params_subtree(variables):
    if isinstance(variables, dict) or hasattr(variables, "keys"):
        return variables["params"] if "params" in variables else variables
    return variables


def _assert_tree_shapes_match(lhs, rhs, lhs_name, rhs_name):
    lhs_flat = traverse_util.flatten_dict(unfreeze(lhs))
    rhs_flat = traverse_util.flatten_dict(unfreeze(rhs))
    if set(lhs_flat) != set(rhs_flat):
        missing_lhs = sorted(set(rhs_flat) - set(lhs_flat))
        missing_rhs = sorted(set(lhs_flat) - set(rhs_flat))
        raise ValueError(
            f"{lhs_name} and {rhs_name} parameter keys do not match. "
            f"Missing in {lhs_name}: {missing_lhs}; missing in {rhs_name}: {missing_rhs}"
        )
    for key in lhs_flat:
        if lhs_flat[key].shape != rhs_flat[key].shape:
            raise ValueError(
                f"Pretrained encoder shape mismatch at {key}: "
                f"{lhs_name} has {lhs_flat[key].shape}, {rhs_name} has {rhs_flat[key].shape}."
            )


def copy_single_encoder_to_two_stream(two_stream_params, single_params):
    params = unfreeze(two_stream_params)
    target = params["params"] if "params" in params else params
    single = unfreeze(_params_subtree(single_params))

    _assert_tree_shapes_match(target["self_cnn"], single["CNN_0"], "self_cnn", "CNN_0")
    _assert_tree_shapes_match(target["other_cnn"], single["CNN_0"], "other_cnn", "CNN_0")
    _assert_tree_shapes_match(target["self_ln"], single["LayerNorm_0"], "self_ln", "LayerNorm_0")
    _assert_tree_shapes_match(target["other_ln"], single["LayerNorm_0"], "other_ln", "LayerNorm_0")

    target["self_cnn"] = single["CNN_0"]
    target["other_cnn"] = single["CNN_0"]
    target["self_ln"] = single["LayerNorm_0"]
    target["other_ln"] = single["LayerNorm_0"]

    if "params" in params:
        params["params"] = target
    return freeze(params)


def build_trainable_labels(params, config):
    trainable_paths = {
        ("params", "fusion_rnn"),
        ("params", "actor_fc"),
        ("params", "actor_out"),
        ("params", "critic_fc"),
        ("params", "critic_out"),
    }
    if config.get("USE_SELF_PRED", True):
        trainable_paths.update({("params", "self_pred_fc"), ("params", "self_pred_out")})

    if config.get("FINETUNE_SELF_STREAM", True):
        trainable_paths.update({("params", "self_cnn"), ("params", "self_ln")})
    if config.get("FINETUNE_OTHER_STREAM", False):
        trainable_paths.update({("params", "other_cnn"), ("params", "other_ln")})

    flat_params = traverse_util.flatten_dict(unfreeze(params))
    flat_labels = {}
    for key in flat_params:
        flat_labels[key] = (
            "train"
            if any(key[: len(path)] == path for path in trainable_paths)
            else "freeze"
        )
    return freeze(traverse_util.unflatten_dict(flat_labels))


def make_train(config, pretrained_params):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if not config.get("PARAMETER_SHARING", True):
        raise NotImplementedError("ippo_lm.py currently supports PARAMETER_SHARING=True.")

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = TwoStreamActorCriticRNN(env.action_space().n, config=config)
        fusion_hidden_dim = 2 * config["GRU_HIDDEN_DIM"]

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *env.observation_space()[0].shape)),
            jnp.zeros((1, config["NUM_ACTORS"]), dtype=bool),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], fusion_hidden_dim
        )
        network_params = network.init(_rng, init_hstate, init_x)
        network_params = copy_single_encoder_to_two_stream(
            network_params, pretrained_params
        )

        if config["ANNEAL_LR"]:
            base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        tx = optax.multi_transform(
            {"train": base_tx, "freeze": optax.set_to_zero()},
            build_trainable_labels(network_params, config),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], fusion_hidden_dim
        )

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    rng,
                ) = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = batchify_obs(last_obs)
                ac_in = (obs_batch[None, :], last_done[None, :])
                hstate, pi, value, _ = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze(axis=0)
                log_prob = pi.log_prob(action).squeeze(axis=0)
                value = value.squeeze(axis=0)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = [v.flatten() for v in env_act.values()]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify_done(done, env.agents, config["NUM_ACTORS"])
                transition = Transition(
                    done_batch,
                    action,
                    value,
                    batchify_reward(reward, env.agents, config["NUM_ACTORS"]),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )
            last_obs_batch = batchify_obs(last_obs)
            ac_in = (last_obs_batch[None, :], last_done[None, :])
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(axis=0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1 - transition.done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - transition.done)
                        * gae
                    )
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value, aux = network.apply(
                            params,
                            init_hstate.squeeze(axis=0),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        approx_kl = (traj_batch.log_prob - log_prob).mean()
                        clip_frac = (
                            jnp.abs(ratio - 1.0) > config["CLIP_EPS"]
                        ).astype(jnp.float32).mean()
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        pred_loss = jnp.asarray(0.0, dtype=value.dtype)
                        if config.get("USE_SELF_PRED", True):
                            hidden = aux["rnn_hidden"]
                            pred_hidden_repr = aux["pred_hidden_repr"]
                            next_hidden = jnp.concatenate(
                                [hidden[1:], jnp.zeros_like(hidden[-1:])], axis=0
                            )
                            next_pred_hidden_repr = jnp.concatenate(
                                [
                                    pred_hidden_repr[1:],
                                    jnp.zeros_like(pred_hidden_repr[-1:]),
                                ],
                                axis=0,
                            )
                            not_done = 1.0 - traj_batch.done.astype(jnp.float32)
                            has_next_step = jnp.ones_like(not_done).at[-1].set(0.0)
                            future_mask = (not_done * has_next_step)[..., None, None]
                            pred_gammas = jnp.asarray(
                                config.get("SELF_PRED_GAMMAS", (0.0, 0.5, 0.9)),
                                dtype=hidden.dtype,
                            ).reshape((1, 1, -1, 1))
                            pred_target = jax.lax.stop_gradient(
                                future_mask
                                * (
                                    next_hidden[..., None, :]
                                    + pred_gammas * next_pred_hidden_repr
                                )
                            )
                            pred_error_clip = config.get("SELF_PRED_ERROR_CLIP", 10.0)
                            pred_error = jnp.clip(
                                pred_hidden_repr - pred_target,
                                -pred_error_clip,
                                pred_error_clip,
                            )
                            pred_delta = config.get("SELF_PRED_HUBER_DELTA", 1.0)
                            pred_loss = optax.huber_loss(
                                pred_error, jnp.zeros_like(pred_error), delta=pred_delta
                            ).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config.get("SELF_PRED_COEF", 0.1) * pred_loss
                        )
                        metrics = {
                            "loss_total": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "self_pred_loss": pred_loss,
                            "approx_kl": approx_kl,
                            "clip_frac": clip_frac,
                        }
                        return total_loss, metrics

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (_, loss_metrics), grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_metrics

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                assert (
                    config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] == 0
                ), "recurrent PPO minibatches whole actor sequences; NUM_MINIBATCHES must divide NUM_ACTORS"
                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (init_hstate, traj_batch, advantages, targets)
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(axis=0),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            for key, value in loss_info.items():
                metric[f"ppo/{key}"] = value.mean()
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def evaluate(params, env, config):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    hstate = ScannedRNN.initialize_carry(env.num_agents, 2 * config["GRU_HIDDEN_DIM"])
    done_batch = jnp.zeros((env.num_agents,), dtype=bool)

    network = TwoStreamActorCriticRNN(action_dim=env.action_space().n, config=config)
    for _ in range(config["GIF_NUM_FRAMES"]):
        obs_batch = obs.reshape((-1,) + env.observation_space()[0].shape)
        ac_in = (obs_batch[None, :], done_batch[None, :])
        hstate, pi, _, _ = network.apply(params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng).squeeze(axis=0)
        env_act = [int(a) for a in np.array(actions)]

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, env_act)
        done_batch = jnp.array([_agent_value(done, a) for a in env.agents])


def single_run(config, run_name="ippo_lm"):
    config = OmegaConf.to_container(config)
    pretrained_path = config.get("PRETRAINED_PARAMS_PATH")
    if not pretrained_path:
        raise ValueError("ippo_lm.py requires PRETRAINED_PARAMS_PATH.")
    pretrained_params = load_params(pretrained_path)
    exp_name = experiment_name(config, run_name)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "LM", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
        dir=str(REPO_ROOT),
        name=exp_name,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config, pretrained_params))
    out = jax.vmap(train_jit)(rngs)

    filename = f"{config['ENV_NAME']}_{exp_name}_seed{config['SEED']}"
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = repo_path("checkpoints", "individual", f"{filename}.pkl")
    save_params(train_state, save_path)


def tune(default_config, run_name="ippo_lm"):
    import copy

    default_config = OmegaConf.to_container(default_config)
    pretrained_path = default_config.get("PRETRAINED_PARAMS_PATH")
    if not pretrained_path:
        raise ValueError("ippo_lm.py requires PRETRAINED_PARAMS_PATH.")
    pretrained_params = load_params(pretrained_path)

    sweep_config = {
        "name": run_name,
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "SEED": {"values": [42, 52, 62]},
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"], dir=str(REPO_ROOT))
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        wandb.run.name = f"sweep_{experiment_name(config, run_name)}_seed{config['SEED']}"
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, pretrained_params)))
        jax.block_until_ready(train_vjit(rngs))

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def main_from_config(config, run_name="ippo_lm"):
    if config["TUNE"]:
        tune(config, run_name)
    else:
        single_run(config, run_name)


@hydra.main(version_base=None, config_path="config", config_name="ippo_lm")
def main(config):
    main_from_config(config, "ippo_lm")


if __name__ == "__main__":
    main()
