"""SVO recurrent PPO with optional partner hidden-state input."""

import copy
import functools
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
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from PIL import Image
from socialjax.wrappers.baselines import SVOLogWrapper

from algorithms.IPPO.ippo_rnn import (
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


def repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)


def mode_name(config):
    return "tom" if config.get("TOM_AUX_HIDDEN", False) else "vanilla"


class ActorCriticSVORNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        assert obs.ndim == 5, f"Expected obs (T,B,H,W,C), got {obs.shape}"

        h, w, c = obs.shape[-3:]
        flat_obs = obs.reshape(-1, h, w, c)
        embedding = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="cnn",
        )(flat_obs)
        embedding = embedding.reshape(*obs.shape[:-3], -1)
        embedding = nn.LayerNorm(name="ln")(embedding)

        if self.config.get("TOM_AUX_HIDDEN", False):
            hidden, embedding = PartnerHiddenScannedRNN(
                num_agents=int(self.config.get("NUM_AGENTS", 2)),
                stop_grad_partner=self.config.get("TOM_STOP_GRAD_PARTNER", True),
                name="rnn",
            )(hidden, (embedding, dones))
        else:
            hidden, embedding = ScannedRNN(name="rnn")(hidden, (embedding, dones))

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
        return hidden, pi, jnp.squeeze(critic, axis=-1)


class PartnerHiddenScannedRNN(nn.Module):
    num_agents: int = 2
    stop_grad_partner: bool = True

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x

        if self.num_agents != 2:
            raise ValueError("PartnerHiddenScannedRNN currently supports exactly two agents.")
        if ins.shape[0] % self.num_agents != 0:
            raise ValueError(
                f"Actor batch {ins.shape[0]} is not divisible by num_agents={self.num_agents}."
            )

        hidden_size = rnn_state.shape[-1]
        new_carry = ScannedRNN.initialize_carry(ins.shape[0], hidden_size)
        rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)

        num_envs = ins.shape[0] // self.num_agents
        partner_state = rnn_state.reshape(
            self.num_agents, num_envs, hidden_size
        )
        partner_state = jnp.flip(partner_state, axis=0).reshape(rnn_state.shape)
        if self.stop_grad_partner:
            partner_state = jax.lax.stop_gradient(partner_state)

        rnn_input = jnp.concatenate([ins, partner_state], axis=-1)
        new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, rnn_input)
        return new_rnn_state, y


def _env_pair_minibatches(batch, permutation, num_agents, num_envs, num_minibatches):
    envs_per_minibatch = num_envs // num_minibatches

    def reshape_one(x):
        x = x.reshape((x.shape[0], num_agents, num_envs) + x.shape[2:])
        x = jnp.take(x, permutation, axis=2)
        x = x.reshape(
            (x.shape[0], num_agents, num_minibatches, envs_per_minibatch)
            + x.shape[3:]
        )
        x = jnp.moveaxis(x, 2, 0)
        return x.reshape(
            (num_minibatches, x.shape[1], num_agents * envs_per_minibatch)
            + x.shape[4:]
        )

    return jax.tree_util.tree_map(reshape_one, batch)


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if not config.get("PARAMETER_SHARING", True):
        raise NotImplementedError("svo_rnn.py currently supports PARAMETER_SHARING=True.")
    if env.num_agents != 2:
        raise ValueError("svo_rnn.py currently expects exactly two agents.")

    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = SVOLogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticSVORNN(env.action_space().n, config=config)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *env.observation_space()[0].shape)),
            jnp.zeros((1, config["NUM_ACTORS"]), dtype=bool),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        network_params = network.init(_rng, init_hstate, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
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
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
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
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
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
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
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
                        _, pi, value = network.apply(
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
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        metrics = {
                            "loss_total": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
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
                    config["NUM_ENVS"] % config["NUM_MINIBATCHES"] == 0
                ), "paired recurrent PPO minibatches require NUM_MINIBATCHES to divide NUM_ENVS"
                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (init_hstate, traj_batch, advantages, targets)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                minibatches = _env_pair_minibatches(
                    batch,
                    permutation,
                    env.num_agents,
                    config["NUM_ENVS"],
                    config["NUM_MINIBATCHES"],
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
            metric["tom_aux_hidden"] = jnp.asarray(
                config.get("TOM_AUX_HIDDEN", False), dtype=jnp.float32
            )
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


def evaluate(params, env, config, run_name="svo_rnn"):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    hstate = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
    done_batch = jnp.zeros((env.num_agents,), dtype=bool)
    pics = [env.render(state)]

    network = ActorCriticSVORNN(action_dim=env.action_space().n, config=config)
    for _ in range(config["GIF_NUM_FRAMES"]):
        obs_batch = obs.reshape((-1,) + env.observation_space()[0].shape)
        ac_in = (obs_batch[None, :], done_batch[None, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng).squeeze(axis=0)
        env_act = [int(a) for a in np.array(actions)]

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, env_act)
        done_batch = jnp.array([_agent_value(done, a) for a in env.agents])
        pics.append(env.render(state))

    root_dir = repo_path("evaluation", config["ENV_NAME"])
    root_dir.mkdir(parents=True, exist_ok=True)
    pics = [Image.fromarray(np.array(img)) for img in pics]
    gif_path = root_dir / (
        f"{run_name}_{env.num_agents}-agents_seed-{config['SEED']}_frames-{config['GIF_NUM_FRAMES']}.gif"
    )
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )
    wandb.log(
        {"Episode GIF": wandb.Video(str(gif_path), caption="Evaluation Episode", format="gif")}
    )


def single_run(config, run_name="svo_rnn"):
    config = OmegaConf.to_container(config)
    config["NUM_AGENTS"] = int(config["ENV_KWARGS"]["num_agents"])
    exp_name = f"{run_name}_{mode_name(config)}_{config['ENV_NAME']}"

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["SVO", "RNN", mode_name(config), config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
        dir=str(REPO_ROOT),
        name=exp_name,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    filename = f"{config['ENV_NAME']}_{run_name}_{mode_name(config)}_seed{config['SEED']}"
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = repo_path("checkpoints", "svo", f"{filename}.pkl")
    save_params(train_state, save_path)
    if not config.get("EVALUATE", True):
        return
    params = load_params(save_path)
    evaluate(
        params,
        socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
        config,
        run_name=f"{run_name}_{mode_name(config)}",
    )


def tune(default_config, run_name="svo_rnn"):
    default_config = OmegaConf.to_container(default_config)
    sweep_config = {
        "name": run_name,
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
            "goal": "maximize",
        },
        "parameters": {
            "SEED": {"values": [42, 52, 62]},
            "TOM_AUX_HIDDEN": {"values": [False, True]},
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

        wandb.run.name = (
            f"sweep_{config['ENV_NAME']}_{run_name}_{mode_name(config)}_seed{config['SEED']}"
        )
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        jax.block_until_ready(train_vjit(rngs))

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def main_from_config(config, run_name="svo_rnn"):
    if config["TUNE"]:
        tune(config, run_name)
    else:
        single_run(config, run_name)


@hydra.main(version_base=None, config_path="config", config_name="svo_rnn")
def main(config):
    main_from_config(config, "svo_rnn")


if __name__ == "__main__":
    main()
