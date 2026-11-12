"""Learners for exposure_env_v1 (NumPy only).

RL learners (count toward the registered three):
  * PPO   - on-policy simulator learner (clipped surrogate, GAE).
  * TD3   - off-policy simulator learner (twin critics, delayed smoothed targets).
  * IQL   - fixed-dataset offline learner (expectile value, AWR extraction).
Non-RL comparators: behaviour cloning and a myopic contextual bandit.
Evaluation: fitted Q evaluation (FQE).

The action seen by every learner is the clipped raw proposal in [-1, 1]. The
shield is part of the environment transition, so rewards, bootstrap targets and
log-probabilities all refer to the same proposal; nothing is relabelled with
the projected action.
"""

from __future__ import annotations

import numpy as np

from .nn import MLP, Adam, polyak

GAMMA = 0.99  # per 6h bar: ~100-bar (25-day) effective horizon
HIDDEN = [64, 64]
LOG_2PI = float(np.log(2.0 * np.pi))


class Policy:
    """Deterministic decoding of an actor network to a raw proposal in [-1, 1]."""

    def __init__(self, actor: MLP, name: str, obs_mask=None):
        self.actor, self.name = actor, name
        self.obs_mask = None if obs_mask is None else np.asarray(obs_mask, dtype=np.float64)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mask is not None:
            obs = obs * self.obs_mask
        return np.clip(self.actor.forward(obs, cache=False)[:, 0], -1.0, 1.0)


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------
def gae(rewards, values, next_values, cut, gamma=GAMMA, lam=0.95):
    """Generalised advantage estimation. ``cut[t]`` ends the trace after step t.

    ``next_values[t]`` must already be zero where the step terminated, and the
    value of the true final observation where it was truncated.
    """
    steps = len(rewards)
    advantages = np.zeros_like(rewards)
    trace = np.zeros_like(rewards[0])
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        trace = delta + gamma * lam * np.where(cut[t], 0.0, trace)
        advantages[t] = trace
    return advantages, advantages + values


def ppo_loss_and_grads(actor: MLP, log_std: np.ndarray, obs, actions, old_logp, advantages, clip=0.2, entropy_coef=0.0):
    """Clipped-surrogate loss (to minimise) and its exact gradients."""
    mean = actor.forward(obs)[:, 0]
    std = np.exp(log_std[0])
    z = (actions - mean) / std
    logp = -0.5 * z * z - log_std[0] - 0.5 * LOG_2PI
    ratio = np.exp(logp - old_logp)
    clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip)
    unclipped_term, clipped_term = ratio * advantages, clipped * advantages
    use_unclipped = unclipped_term <= clipped_term
    count = len(obs)
    entropy = log_std[0] + 0.5 * (1.0 + LOG_2PI)
    loss = -np.mean(np.minimum(unclipped_term, clipped_term)) - entropy_coef * entropy
    dlogp = np.where(use_unclipped, -advantages * ratio, 0.0) / count
    grads, _ = actor.backward((dlogp * z / std)[:, None])
    grad_log_std = np.array([np.sum(dlogp * (z * z - 1.0)) - entropy_coef])
    return float(loss), grads, grad_log_std, float(np.mean(old_logp - logp))


def train_ppo(env, seed: int, total_steps: int, rollout=128, epochs=4, minibatch=1024, lr=3e-4, lam=0.95, clip=0.2,
              entropy_coef=0.001, init_log_std=-0.7, checkpoint=None, checkpoint_every=0, risk_penalty=0.0, obs_mask=None):
    rng = np.random.default_rng(seed)
    obs_dim = env.observe().shape[1]
    actor = MLP([obs_dim, *HIDDEN, 1], rng, out_tanh=True, out_scale=0.1)
    critic = MLP([obs_dim, *HIDDEN, 1], rng)
    log_std = np.array([init_log_std])
    actor_opt = Adam(actor.params + [log_std], lr, max_norm=0.5)
    critic_opt = Adam(critic.params, lr, max_norm=0.5)
    mask = np.ones(obs_dim) if obs_mask is None else np.asarray(obs_mask, dtype=np.float64)
    obs = env.reset() * mask
    lanes = env.lanes
    history, done_steps, next_checkpoint = [], 0, checkpoint_every
    while done_steps < total_steps:
        buf_obs = np.zeros((rollout, lanes, obs_dim))
        buf_act = np.zeros((rollout, lanes))
        buf_logp = np.zeros((rollout, lanes))
        buf_rew = np.zeros((rollout, lanes))
        buf_val = np.zeros((rollout, lanes))
        buf_next = np.zeros((rollout, lanes))
        buf_cut = np.zeros((rollout, lanes), dtype=bool)
        std = float(np.exp(log_std[0]))
        for t in range(rollout):
            mean = actor.forward(obs, cache=False)[:, 0]
            sample = mean + std * rng.standard_normal(lanes)
            buf_obs[t], buf_act[t] = obs, sample
            buf_logp[t] = -0.5 * ((sample - mean) / std) ** 2 - log_std[0] - 0.5 * LOG_2PI
            buf_val[t] = critic.forward(obs, cache=False)[:, 0]
            obs, reward, terminated, truncated, final_obs, _ = env.step(np.clip(sample, -1.0, 1.0))
            obs, final_obs = obs * mask, final_obs * mask
            # Registered shaping (extension only); the economic ledger is untouched.
            buf_rew[t] = reward - risk_penalty * reward * reward
            # Bootstrap from the true successor unless the episode terminated.
            buf_next[t] = np.where(terminated, 0.0, critic.forward(final_obs, cache=False)[:, 0])
            buf_cut[t] = terminated | truncated
        advantages, returns = gae(buf_rew, buf_val, buf_next, buf_cut, GAMMA, lam)
        flat = lambda x: x.reshape(rollout * lanes, *x.shape[2:])
        f_obs, f_act, f_logp, f_adv, f_ret = flat(buf_obs), flat(buf_act), flat(buf_logp), flat(advantages), flat(returns)
        size = len(f_obs)
        kl = 0.0
        for _ in range(epochs):
            order = rng.permutation(size)
            for start in range(0, size, minibatch):
                idx = order[start : start + minibatch]
                adv = f_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                _, grads, grad_std, kl = ppo_loss_and_grads(actor, log_std, f_obs[idx], f_act[idx], f_logp[idx], adv, clip, entropy_coef)
                actor_opt.step(grads + [grad_std])
                log_std[0] = np.clip(log_std[0], -3.0, 0.5)
                value = critic.forward(f_obs[idx])[:, 0]
                v_grads, _ = critic.backward(((value - f_ret[idx]) / len(idx))[:, None])
                critic_opt.step(v_grads)
        done_steps += size
        history.append({"steps": done_steps, "meanReward": float(buf_rew.mean()), "logStd": float(log_std[0]), "approxKl": kl})
        if checkpoint is not None and checkpoint_every and done_steps >= next_checkpoint:
            checkpoint(done_steps, Policy(actor, "ppo", obs_mask))
            next_checkpoint += checkpoint_every
    return Policy(actor, "ppo", obs_mask), {"history": history, "critic": critic, "logStd": float(log_std[0])}


# --------------------------------------------------------------------------
# Shared critic helpers
# --------------------------------------------------------------------------
def _q_forward(critic: MLP, obs, actions, params=None, cache=True):
    return critic.forward(np.concatenate([obs, actions[:, None]], axis=1), params=params, cache=cache)[:, 0]


def _q_fit_step(critic: MLP, opt: Adam, obs, actions, targets) -> float:
    q = _q_forward(critic, obs, actions)
    error = q - targets
    grads, _ = critic.backward((error / len(obs))[:, None])
    opt.step(grads)
    return float(np.mean(error * error))


class Replay:
    def __init__(self, capacity: int, obs_dim: int):
        self.obs = np.zeros((capacity, obs_dim))
        self.next_obs = np.zeros((capacity, obs_dim))
        self.act = np.zeros(capacity)
        self.rew = np.zeros(capacity)
        self.term = np.zeros(capacity, dtype=bool)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, obs, act, rew, next_obs, term) -> None:
        for row in range(len(obs)):
            i = self.cursor
            self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.term[i] = obs[row], act[row], rew[row], next_obs[row], term[row]
            self.cursor = (self.cursor + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch: int):
        idx = rng.integers(0, self.size, size=batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.term[idx]


# --------------------------------------------------------------------------
# TD3
# --------------------------------------------------------------------------
def td3_target(reward, terminated, next_q1, next_q2, gamma=GAMMA):
    """Clipped double-Q target; truncation is not termination and still bootstraps."""
    return reward + gamma * np.where(terminated, 0.0, np.minimum(next_q1, next_q2))


def train_td3(env, seed: int, total_steps: int, warmup=5_000, batch=256, updates_per_step=4, lr=3e-4, tau=0.005,
              explore_noise=0.2, target_noise=0.2, noise_clip=0.5, policy_delay=2, capacity=300_000,
              checkpoint=None, checkpoint_every=0):
    rng = np.random.default_rng(seed)
    obs_dim = env.observe().shape[1]
    actor = MLP([obs_dim, *HIDDEN, 1], rng, out_tanh=True, out_scale=0.1)
    q1 = MLP([obs_dim + 1, *HIDDEN, 1], rng)
    q2 = MLP([obs_dim + 1, *HIDDEN, 1], rng)
    actor_t, q1_t, q2_t = actor.copy_params(), q1.copy_params(), q2.copy_params()
    actor_opt, q1_opt, q2_opt = Adam(actor.params, lr, max_norm=1.0), Adam(q1.params, lr, max_norm=10.0), Adam(q2.params, lr, max_norm=10.0)
    replay = Replay(capacity, obs_dim)
    obs = env.reset()
    lanes = env.lanes
    history, steps, updates, next_checkpoint = [], 0, 0, checkpoint_every
    q_loss = 0.0
    while steps < total_steps:
        if steps < warmup:
            action = rng.uniform(-1.0, 1.0, lanes)
        else:
            action = np.clip(actor.forward(obs, cache=False)[:, 0] + explore_noise * rng.standard_normal(lanes), -1.0, 1.0)
        next_obs, reward, terminated, _, final_obs, _ = env.step(action)
        replay.add(obs, action, reward, final_obs, terminated)
        obs = next_obs
        steps += lanes
        if steps >= warmup:
            for _ in range(updates_per_step):
                b_obs, b_act, b_rew, b_next, b_term = replay.sample(rng, batch)
                noise = np.clip(target_noise * rng.standard_normal(batch), -noise_clip, noise_clip)
                next_action = np.clip(actor.forward(b_next, params=actor_t, cache=False)[:, 0] + noise, -1.0, 1.0)
                target = td3_target(b_rew, b_term, _q_forward(q1, b_next, next_action, q1_t, False), _q_forward(q2, b_next, next_action, q2_t, False))
                q_loss = _q_fit_step(q1, q1_opt, b_obs, b_act, target)
                _q_fit_step(q2, q2_opt, b_obs, b_act, target)
                updates += 1
                if updates % policy_delay == 0:
                    pi = actor.forward(b_obs)[:, 0]
                    _q_forward(q1, b_obs, pi)
                    _, grad_in = q1.backward(np.full((batch, 1), -1.0 / batch))
                    a_grads, _ = actor.backward(grad_in[:, -1:])
                    actor_opt.step(a_grads)
                    polyak(actor_t, actor.params, tau)
                    polyak(q1_t, q1.params, tau)
                    polyak(q2_t, q2.params, tau)
        if steps % (lanes * 250) == 0:
            history.append({"steps": steps, "updates": updates, "qLoss": q_loss})
        if checkpoint is not None and checkpoint_every and steps >= next_checkpoint:
            checkpoint(steps, Policy(actor, "td3"))
            next_checkpoint += checkpoint_every
    return Policy(actor, "td3"), {"history": history, "q1": q1, "q2": q2, "updates": updates}


# --------------------------------------------------------------------------
# Offline: IQL, behaviour cloning, contextual bandit, FQE
# --------------------------------------------------------------------------
def expectile_grad(diff: np.ndarray, expectile: float) -> np.ndarray:
    """d/dV of mean(|expectile - 1(diff<0)| * diff^2) with diff = Q - V."""
    weight = np.where(diff < 0, 1.0 - expectile, expectile)
    return -2.0 * weight * diff / len(diff)


def train_iql(data: dict, seed: int, updates: int, batch=256, lr=3e-4, expectile=0.7, beta=30.0, weight_clip=100.0, tau=0.005,
              checkpoint=None, checkpoint_every=0):
    rng = np.random.default_rng(seed)
    obs_dim = data["obs"].shape[1]
    actor = MLP([obs_dim, *HIDDEN, 1], rng, out_tanh=True, out_scale=0.1)
    value = MLP([obs_dim, *HIDDEN, 1], rng)
    q1 = MLP([obs_dim + 1, *HIDDEN, 1], rng)
    q2 = MLP([obs_dim + 1, *HIDDEN, 1], rng)
    q1_t, q2_t = q1.copy_params(), q2.copy_params()
    opts = [Adam(net.params, lr, max_norm=10.0) for net in (actor, value, q1, q2)]
    size = len(data["obs"])
    history = []
    for update in range(1, updates + 1):
        idx = rng.integers(0, size, size=batch)
        obs, act, rew, nxt, term = data["obs"][idx], data["act"][idx], data["rew"][idx], data["next_obs"][idx], data["term"][idx]
        target_q = np.minimum(_q_forward(q1, obs, act, q1_t, False), _q_forward(q2, obs, act, q2_t, False))
        v = value.forward(obs)[:, 0]
        v_grads, _ = value.backward(expectile_grad(target_q - v, expectile)[:, None])
        opts[1].step(v_grads)
        backup = rew + GAMMA * np.where(term, 0.0, value.forward(nxt, cache=False)[:, 0])
        q_loss = _q_fit_step(q1, opts[2], obs, act, backup)
        _q_fit_step(q2, opts[3], obs, act, backup)
        weights = np.minimum(np.exp(beta * (target_q - v)), weight_clip)
        mean = actor.forward(obs)[:, 0]
        a_grads, _ = actor.backward((2.0 * weights * (mean - act) / batch)[:, None])
        opts[0].step(a_grads)
        polyak(q1_t, q1.params, tau)
        polyak(q2_t, q2.params, tau)
        if update % 2000 == 0:
            history.append({"updates": update, "qLoss": q_loss, "meanWeight": float(weights.mean()), "clippedWeights": float(np.mean(weights >= weight_clip))})
        if checkpoint is not None and checkpoint_every and update % checkpoint_every == 0:
            checkpoint(update, Policy(actor, "iql"))
    return Policy(actor, "iql"), {"history": history, "q1": q1, "q2": q2, "value": value}


def train_bc(data: dict, seed: int, updates: int, batch=256, lr=3e-4):
    """Behaviour cloning: mean-squared regression onto logged proposals. Not RL."""
    rng = np.random.default_rng(seed)
    actor = MLP([data["obs"].shape[1], *HIDDEN, 1], rng, out_tanh=True, out_scale=0.1)
    opt = Adam(actor.params, lr, max_norm=10.0)
    for _ in range(updates):
        idx = rng.integers(0, len(data["obs"]), size=batch)
        mean = actor.forward(data["obs"][idx])[:, 0]
        grads, _ = actor.backward((2.0 * (mean - data["act"][idx]) / batch)[:, None])
        opt.step(grads)
    return Policy(actor, "behavior_cloning"), {}


class BanditPolicy:
    """Myopic contextual bandit: greedy over a one-step reward model. Not RL."""

    name = "contextual_bandit"
    GRID = np.linspace(-1.0, 1.0, 21)

    def __init__(self, model: MLP):
        self.model = model

    def act(self, obs: np.ndarray) -> np.ndarray:
        scores = np.stack([_q_forward(self.model, obs, np.full(len(obs), a), cache=False) for a in self.GRID], axis=1)
        return self.GRID[np.argmax(scores, axis=1)]


def train_bandit(data: dict, seed: int, updates: int, batch=256, lr=3e-4):
    rng = np.random.default_rng(seed)
    model = MLP([data["obs"].shape[1] + 1, *HIDDEN, 1], rng)
    opt = Adam(model.params, lr, max_norm=10.0)
    for _ in range(updates):
        idx = rng.integers(0, len(data["obs"]), size=batch)
        _q_fit_step(model, opt, data["obs"][idx], data["act"][idx], data["rew"][idx])
    return BanditPolicy(model), {}


def fitted_q_evaluation(data: dict, policy, seed: int, updates: int, batch=256, lr=3e-4, tau=0.005):
    """FQE of a deterministic policy on a fixed dataset. Returns the fitted critic."""
    rng = np.random.default_rng(seed)
    obs_dim = data["obs"].shape[1]
    q = MLP([obs_dim + 1, *HIDDEN, 1], rng)
    q_t = q.copy_params()
    opt = Adam(q.params, lr, max_norm=10.0)
    next_action = policy.act(data["next_obs"])
    for _ in range(updates):
        idx = rng.integers(0, len(data["obs"]), size=batch)
        nxt_q = _q_forward(q, data["next_obs"][idx], next_action[idx], q_t, False)
        target = data["rew"][idx] + GAMMA * np.where(data["term"][idx], 0.0, nxt_q)
        _q_fit_step(q, opt, data["obs"][idx], data["act"][idx], target)
        polyak(q_t, q.params, tau)
    return q


def q_value(critic: MLP, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return _q_forward(critic, obs, actions, cache=False)
