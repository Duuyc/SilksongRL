import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from frame_encoder import FrameStackFeatureExtractor


class RamNormalizeWrapper(gym.ObservationWrapper):
    """Convert Atari RAM uint8 observations to normalized float32 vectors."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32) / 255.0


class ActionSubsetWrapper(gym.ActionWrapper):
    """Restrict Pong to a small, interpretable action subset."""

    def __init__(self, env: gym.Env, action_indices: list[int]):
        super().__init__(env)
        self.action_indices = list(action_indices)
        self.action_space = spaces.Discrete(len(self.action_indices))

    def action(self, action: int) -> int:
        return self.action_indices[int(action)]


def parse_action_subset(name: str) -> list[int] | None:
    key = (name or "all").strip().lower().replace("-", "_")
    if key in ("all", "full", "none"):
        return None
    if key in ("noop_right_left", "3", "minimal3"):
        return [0, 2, 3]
    if key in ("right_left", "2", "minimal2"):
        return [2, 3]
    raise ValueError(f"Unknown action subset: {name}")


def register_ale() -> None:
    import ale_py

    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)


def make_pong_ram_env(seed: int, rank: int, output_dir: Path, action_subset: str = "all"):
    def _init():
        register_ale()
        try:
            env = gym.make(
                "ALE/Pong-ram-v5",
                frameskip=4,
                repeat_action_probability=0.25,
                full_action_space=False,
            )
        except gym.error.NameNotFound:
            env = gym.make(
                "ALE/Pong-v5",
                obs_type="ram",
                frameskip=4,
                repeat_action_probability=0.25,
                full_action_space=False,
            )
        indices = parse_action_subset(action_subset)
        if indices is not None:
            env = ActionSubsetWrapper(env, indices)
        env = RamNormalizeWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_eval_env(seed: int, output_dir: Path, action_subset: str, frame_stack: int):
    env = DummyVecEnv([make_pong_ram_env(seed, 0, output_dir, action_subset=action_subset)])
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


class EvalAndPlotCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        output_dir: Path,
        eval_freq: int = 25_000,
        eval_episodes: int = 10,
        initial_history: list[dict[str, Any]] | None = None,
        initial_best_mean_reward: float | None = None,
        early_stop_score: float | None = None,
        early_stop_patience_evals: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.output_dir = output_dir
        self.eval_freq = max(1, eval_freq)
        self.eval_episodes = max(1, eval_episodes)
        self.best_mean_reward = (
            float(initial_best_mean_reward)
            if initial_best_mean_reward is not None
            else -float("inf")
        )
        self.history: list[dict[str, Any]] = list(initial_history or [])
        self.early_stop_score = early_stop_score
        self.early_stop_patience_evals = max(0, int(early_stop_patience_evals))
        self.early_stop_hits = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.eval_freq or self.num_timesteps % self.eval_freq != 0:
            return True

        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
        )
        rewards = np.asarray(rewards, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.float32)
        mean_reward = float(rewards.mean())
        std_reward = float(rewards.std())
        full_score_rate = float(np.mean(rewards >= 20.0))
        positive_rate = float(np.mean(rewards > 0.0))
        row = {
            "timesteps": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": float(lengths.mean()),
            "positive_rate": positive_rate,
            "full_score_rate": full_score_rate,
        }
        self.history.append(row)
        save_json(self.output_dir / "eval_history.json", self.history)
        plot_eval_history(self.output_dir / "eval_curve.png", self.history)

        if self.verbose:
            print(
                "[PongRAM] eval "
                f"steps={self.num_timesteps}, mean={mean_reward:.2f}, "
                f"std={std_reward:.2f}, positive={positive_rate:.1%}, "
                f"full_score={full_score_rate:.1%}",
                flush=True,
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.output_dir / "best_model.zip"))
            if self.verbose:
                print(f"[PongRAM] saved best_model.zip, mean_reward={mean_reward:.2f}", flush=True)

        if self.early_stop_score is not None and mean_reward >= self.early_stop_score:
            self.early_stop_hits += 1
            if self.verbose:
                print(
                    f"[PongRAM] early-stop hit {self.early_stop_hits}/"
                    f"{max(1, self.early_stop_patience_evals)} at {mean_reward:.2f}",
                    flush=True,
                )
            if self.early_stop_hits >= max(1, self.early_stop_patience_evals):
                return False
        else:
            self.early_stop_hits = 0

        return True


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_eval_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    x = [h["timesteps"] for h in history]
    mean_reward = [h["mean_reward"] for h in history]
    positive_rate = [100.0 * h["positive_rate"] for h in history]
    full_score_rate = [100.0 * h["full_score_rate"] for h in history]

    plt.figure(figsize=(10, 5), dpi=160)
    ax1 = plt.gca()
    ax1.plot(x, mean_reward, color="#f26b38", marker="o", label="Mean episode score")
    ax1.axhline(20.0, color="#94a3b8", linestyle="--", linewidth=1, label="near full score")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Pong score")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, positive_rate, color="#35b8a5", marker="s", label="Positive-score rate")
    ax2.plot(x, full_score_rate, color="#22324d", marker="^", label="Full-score rate")
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 100)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    plt.title("ALE/Pong RAM PPO benchmark")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO benchmark on non-visual ALE/Pong RAM observations.")
    parser.add_argument("--total-timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--use-frame-encoder", action="store_true")
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--action-subset", default="all", choices=["all", "noop_right_left", "right_left"])
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume from latest_model.zip/final_model.zip if present.")
    parser.add_argument("--early-stop-score", type=float, default=None)
    parser.add_argument("--early-stop-patience-evals", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "pong_ram_ppo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.total_timesteps = min(args.total_timesteps, 5_000)
        args.eval_freq = min(args.eval_freq, 2_500)
        args.eval_episodes = min(args.eval_episodes, 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", vars(args))

    register_ale()
    train_env = DummyVecEnv(
        [make_pong_ram_env(args.seed, i, output_dir, action_subset=args.action_subset) for i in range(args.n_envs)]
    )
    if args.frame_stack > 1:
        train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
    train_env = VecMonitor(train_env)
    eval_env = make_eval_env(args.seed + 10_000, output_dir, args.action_subset, args.frame_stack)

    policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
    if args.use_frame_encoder:
        policy_kwargs = {
            "features_extractor_class": FrameStackFeatureExtractor,
            "features_extractor_kwargs": {
                "frame_stack": max(1, args.frame_stack),
                "per_frame_features": 64,
                "features_dim": args.features_dim,
            },
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        }

    resume_path = None
    if args.resume:
        for candidate in ("latest_model.zip", "final_model.zip", "best_model.zip"):
            path = output_dir / candidate
            if path.exists():
                resume_path = path
                break

    if resume_path is not None:
        print(f"[PongRAM] Resuming from {resume_path}", flush=True)
        model = PPO.load(str(resume_path), env=train_env, device=args.device)
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=args.seed,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
        )

    history = load_json(output_dir / "eval_history.json", [])
    best_from_history = max((float(row.get("mean_reward", -float("inf"))) for row in history), default=None)

    callback = EvalAndPlotCallback(
        eval_env,
        output_dir=output_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        initial_history=history,
        initial_best_mean_reward=best_from_history,
        early_stop_score=args.early_stop_score,
        early_stop_patience_evals=args.early_stop_patience_evals,
    )

    print(
        "[PongRAM] Training ALE/Pong RAM PPO "
        f"for {args.total_timesteps} steps, n_envs={args.n_envs}, output={output_dir}",
        flush=True,
    )
    started = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=resume_path is None,
    )
    elapsed = time.time() - started
    model.save(str(output_dir / "latest_model.zip"))
    model.save(str(output_dir / "final_model.zip"))

    rewards, lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=max(10, args.eval_episodes),
        deterministic=True,
        return_episode_rewards=True,
        warn=False,
    )
    summary = {
        "total_timesteps": args.total_timesteps,
        "elapsed_seconds": elapsed,
        "final_mean_reward": float(np.mean(rewards)),
        "final_std_reward": float(np.std(rewards)),
        "final_positive_rate": float(np.mean(np.asarray(rewards) > 0.0)),
        "final_full_score_rate": float(np.mean(np.asarray(rewards) >= 20.0)),
        "final_mean_length": float(np.mean(lengths)),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    print("[PongRAM] Final summary:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
