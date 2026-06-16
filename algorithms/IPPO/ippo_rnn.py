"""Task-agnostic recurrent IPPO for CNN SocialJax environments."""

import functools
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Sequence

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
from socialjax.wrappers.baselines import LogWrapper


class ScannedRNN(nn.Module):
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

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])
        rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 4, f"CNN expected (B,H,W,C), got {x.shape}"
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        return x


class ActorCriticRNN(nn.Module):
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
        )(flat_obs)
        embedding = embedding.reshape(*obs.shape[:-3], -1)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def _agent_value(tree, agent):
    return tree[str(agent)] if str(agent) in tree else tree[agent]


def batchify_reward(reward, agent_list, num_actors):
    if isinstance(reward, dict):
        return jnp.stack([_agent_value(reward, a) for a in agent_list]).reshape(
            (num_actors,)
        )
    reward = jnp.asarray(reward)
    if reward.ndim >= 2:
        axes = (1, 0) + tuple(range(2, reward.ndim))
        reward = jnp.transpose(reward, axes)
    return reward.reshape((num_actors,))


def batchify_done(done: dict, agent_list, num_actors):
    return jnp.stack([_agent_value(done, a) for a in agent_list]).reshape(
        (num_actors,)
    )


def batchify_obs(obs):
    obs = jnp.transpose(obs, (1, 0, 2, 3, 4))
    return obs.reshape((-1,) + obs.shape[2:])


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if not config.get("PARAMETER_SHARING", True):
        raise NotImplementedError("ippo_rnn.py currently supports PARAMETER_SHARING=True.")

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
        network = ActorCriticRNN(env.action_space().n, config=config)

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


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
    with open(save_path, "wb") as f:
        pickle.dump(params, f)


def load_params(load_path):
    with open(load_path, "rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def evaluate(params, env, config, run_name="ippo_rnn"):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    hstate = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
    done_batch = jnp.zeros((env.num_agents,), dtype=bool)
    pics = [env.render(state)]

    network = ActorCriticRNN(action_dim=env.action_space().n, config=config)
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

    root_dir = Path("evaluation") / config["ENV_NAME"]
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


def single_run(config, run_name="ippo_rnn"):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{run_name}_{config['ENV_NAME']}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    filename = f"{config['ENV_NAME']}_{run_name}_seed{config['SEED']}"
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/individual/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)
    evaluate(
        params,
        socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
        config,
        run_name=run_name,
    )


def tune(default_config, run_name="ippo_rnn"):
    import copy

    default_config = OmegaConf.to_container(default_config)
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
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        wandb.run.name = f"sweep_{config['ENV_NAME']}_{run_name}_seed{config['SEED']}"
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        jax.block_until_ready(train_vjit(rngs))

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def main_from_config(config, run_name="ippo_rnn"):
    if config["TUNE"]:
        tune(config, run_name)
    else:
        single_run(config, run_name)


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn")
def main(config):
    main_from_config(config, "ippo_rnn")


if __name__ == "__main__":
    main()
