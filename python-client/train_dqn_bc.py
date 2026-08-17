from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Sequence

# This script warms up the DQN Q-network directly. Keep runtime-only replay
# prefill and trace recording out of this offline supervised pass.
os.environ.setdefault("SILKSONGRL_DQN_PREFILL", "0")
os.environ.setdefault("SILKSONGRL_SAVE_AGENT_TRACES", "0")
os.environ.setdefault("SILKSONGRL_FUTURE_PREDICTOR", "0")

import numpy as np
import torch
import torch.nn.functional as F

import dqn_core
from demo_dataset import (
    DEFAULT_DEMO_ROOT,
    LACE_COMPRESSED_DENSE_OBS_SIZE,
    RAW_ACTION_SPACE_SHAPE,
    infer_boss_name,
    load_demonstrations,
)


DEFAULT_OBS_SIZE = int(os.environ.get("SILKSONGRL_DQN_BC_OBS_SIZE", str(LACE_COMPRESSED_DENSE_OBS_SIZE)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior cloning warmup for the DQN Q-network.")
    parser.add_argument("--demo-dir", default=os.path.join(DEFAULT_DEMO_ROOT, "lace_2"))
    parser.add_argument("--boss-name", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--obs-size", type=int, default=DEFAULT_OBS_SIZE)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--attack-weight", type=float, default=3.0)
    parser.add_argument("--tool-weight", type=float, default=1.2)
    parser.add_argument("--bind-weight", type=float, default=2.0)
    parser.add_argument("--jump-weight", type=float, default=1.2)
    parser.add_argument("--dash-weight", type=float, default=1.1)
    return parser.parse_args()


def remove_model_dir(model_key: str) -> None:
    model_root = Path(dqn_core.MODEL_ROOT).resolve()
    model_dir = (model_root / model_key).resolve()
    try:
        model_dir.relative_to(model_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete model directory outside MODEL_ROOT: {model_dir}") from exc
    if model_dir.is_dir():
        shutil.rmtree(model_dir)
        print(f"[DQN-BC] Removed existing model dir: {model_dir}")


def flatten_actions(actions: np.ndarray, action_shape: Sequence[int]) -> np.ndarray:
    return np.asarray([dqn_core.flatten_action(row.tolist(), action_shape) for row in actions], dtype=np.int64)


def sample_weights(actions: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    weights = np.ones((len(actions),), dtype=np.float32)
    if actions.shape[1] >= 8:
        weights *= np.where(actions[:, 3] != 0, args.attack_weight, 1.0)
        weights *= np.where(actions[:, 7] != 0, args.tool_weight, 1.0)
        weights *= np.where(actions[:, 6] != 0, args.bind_weight, 1.0)
        weights *= np.where(actions[:, 2] != 0, args.jump_weight, 1.0)
        weights *= np.where(actions[:, 4] != 0, args.dash_weight, 1.0)
    return weights / max(float(np.mean(weights)), 1e-6)


def evaluate(model: dqn_core.DQNAgent, obs: np.ndarray, actions: np.ndarray, batch_size: int) -> tuple[float, float, np.ndarray]:
    action_shape = model.action_shape
    target_ids = flatten_actions(actions, action_shape)
    losses = []
    exact = []
    branch_correct = np.zeros((len(action_shape),), dtype=np.float64)
    branch_total = 0
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            end = min(start + batch_size, len(obs))
            obs_t = torch.as_tensor(obs[start:end], dtype=torch.float32, device=model.device)
            target_t = torch.as_tensor(target_ids[start:end], dtype=torch.long, device=model.device)
            q = model.online(obs_t)
            losses.append(float(F.cross_entropy(q, target_t).detach().cpu().item()))
            pred_ids = q.argmax(dim=1).cpu().numpy()
            pred_actions = np.asarray([dqn_core.unflatten_action(int(idx), action_shape) for idx in pred_ids], dtype=np.int64)
            action_batch = actions[start:end]
            exact.append(np.all(pred_actions == action_batch, axis=1))
            branch_correct += (pred_actions == action_batch).sum(axis=0)
            branch_total += len(action_batch)
    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.concatenate(exact).mean()) if exact else 0.0,
        branch_correct / max(branch_total, 1),
    )


def train(args: argparse.Namespace) -> None:
    boss_name = args.boss_name or infer_boss_name(args.demo_dir, min_steps=args.min_steps)
    dataset = load_demonstrations(
        demo_dir=args.demo_dir,
        boss_name=boss_name,
        obs_size=args.obs_size,
        action_shape=RAW_ACTION_SPACE_SHAPE,
        min_steps=args.min_steps,
        max_samples=args.max_samples,
        drop_mask_violations=False,
    )

    model_key = dqn_core.model_key_for_action_space(boss_name, RAW_ACTION_SPACE_SHAPE, args.obs_size, 0)
    if args.fresh:
        remove_model_dir(model_key)

    dqn_core.initialize_model(
        args.obs_size,
        boss_name,
        RAW_ACTION_SPACE_SHAPE,
        observation_type="vector",
        vector_obs_size=args.obs_size,
        is_eval=False,
    )
    model = dqn_core.model
    if model is None:
        raise RuntimeError("DQN model failed to initialize")

    for group in model.optimizer.param_groups:
        group["lr"] = args.learning_rate

    observations = dqn_core.augment_observation_batch(dataset.observations).astype(np.float32, copy=False)
    actions = dataset.actions.astype(np.int64, copy=False)
    target_ids = flatten_actions(actions, model.action_shape)
    weights = sample_weights(actions, args)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(observations))
    val_count = max(1, int(len(indices) * max(0.0, min(args.val_fraction, 0.5))))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    print(f"[DQN-BC] Boss: {boss_name}")
    print(f"[DQN-BC] Files: {len(dataset.files)}, samples: {len(observations)}, train={len(train_idx)}, val={len(val_idx)}")
    print(f"[DQN-BC] Model: {model.model_dir}")
    print(
        f"[DQN-BC] Weights: attack={args.attack_weight}, tool={args.tool_weight}, "
        f"bind={args.bind_weight}, jump={args.jump_weight}, dash={args.dash_weight}"
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        epoch_losses = []
        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[start:start + args.batch_size]
            obs_t = torch.as_tensor(observations[batch_idx], dtype=torch.float32, device=model.device)
            target_t = torch.as_tensor(target_ids[batch_idx], dtype=torch.long, device=model.device)
            weight_t = torch.as_tensor(weights[batch_idx], dtype=torch.float32, device=model.device)
            q = model.online(obs_t)
            loss_values = F.cross_entropy(q, target_t, reduction="none")
            loss = (loss_values * weight_t).mean()
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.online.parameters(), 10.0)
            model.optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        val_loss, exact_acc, branch_acc = evaluate(
            model,
            observations[val_idx],
            actions[val_idx],
            args.batch_size,
        )
        print(
            f"[DQN-BC] Epoch {epoch}/{args.epochs}: train_loss={np.mean(epoch_losses):.4f}, "
            f"val_loss={val_loss:.4f}, exact={exact_acc * 100:.1f}%, "
            f"branch_acc=" + ",".join(f"{value * 100:.1f}" for value in branch_acc)
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.target.load_state_dict(model.online.state_dict())
            model.save("best_checkpoint")

    model.target.load_state_dict(model.online.state_dict())
    model.save("checkpoint")
    print("[DQN-BC] Saved checkpoint and best_checkpoint for DQN warm start.")


if __name__ == "__main__":
    train(parse_args())
