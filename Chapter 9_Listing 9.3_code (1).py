# Listing 9.3 — Training loop and policy extraction for the lever environment.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import random
import statistics

# from lever_env import LeverEnv, LeverEnvConfig
# from vdn import VDNTrainer, VDNConfig


@dataclass(frozen=True)
class TrainRunConfig:
    episodes: int = 4000
    log_every: int = 200
    eval_every: int = 500
    eval_episodes: int = 200


def evaluate_greedy(env: "LeverEnv", trainer: "VDNTrainer", episodes: int, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    successes = 0
    steps_to_success = []
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=rng.randrange(1_000_000))
        total_r = 0.0
        for t in range(env.cfg.max_steps):
            o1 = obs[0]
            o2 = obs[1]
            a1 = trainer.q1.argmax(o1, rng)
            a2 = trainer.q2.argmax(o2, rng)
            obs, r, done, info = env.step({0: a1, 1: a2})
            total_r += r
            if done:
                if info.get("success"):
                    successes += 1
                    steps_to_success.append(t + 1)
                break
        returns.append(total_r)

    return {
        "success_rate": successes / max(1, episodes),
        "mean_steps_to_success": statistics.fmean(steps_to_success) if steps_to_success else float("inf"),
        "mean_return": statistics.fmean(returns) if returns else 0.0,
    }


def train_vdn(seed: int = 1) -> None:
    env = LeverEnv(LeverEnvConfig(step_penalty=-0.01), seed=seed)
    eval_env = LeverEnv(LeverEnvConfig(step_penalty=-0.01), seed=seed + 999)
    trainer = VDNTrainer(n_actions=4, cfg=VDNConfig(), seed=seed)
    run_cfg = TrainRunConfig()

    recent_returns = []
    recent_success = []

    for ep in range(1, run_cfg.episodes + 1):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        total_r = 0.0
        success = 0

        for _ in range(env.cfg.max_steps):
            o1 = obs[0]
            o2 = obs[1]
            a1, a2 = trainer.act(o1, o2)
            nobs, r, done, info = env.step({0: a1, 1: a2})

            trainer.push_transition(o1, o2, a1, a2, r, nobs[0], nobs[1], done)
            trainer.train_step()

            obs = nobs
            total_r += r
            if done:
                success = 1 if info.get("success") else 0
                break

        recent_returns.append(total_r)
        recent_success.append(success)
        if len(recent_returns) > run_cfg.log_every:
            recent_returns.pop(0)
            recent_success.pop(0)

        if ep % run_cfg.log_every == 0:
            avg_r = statistics.fmean(recent_returns)
            sr = statistics.fmean(recent_success)
            print(f"ep={ep} epsilon={trainer.epsilon():.3f} avg_return={avg_r:.3f} success_rate={sr:.3f}")

        if ep % run_cfg.eval_every == 0:
            metrics = evaluate_greedy(eval_env, trainer, run_cfg.eval_episodes, seed=seed + ep)
            print(f"[eval] ep={ep} success_rate={metrics['success_rate']:.3f} "
                  f"mean_steps={metrics['mean_steps_to_success']:.2f} mean_return={metrics['mean_return']:.3f}")


if __name__ == "__main__":
    train_vdn()
