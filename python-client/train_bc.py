import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch

import rl_core
from bc_loss import (
    parse_action_value_weights,
    parse_branch_weights,
    parse_nonzero_boosts,
    weighted_multidiscrete_bc_loss,
)
from demo_dataset import (
    COMPOSITE_ACTION_SPACE_SHAPE,
    DEFAULT_DEMO_ROOT,
    LACE_COMPRESSED_DENSE_OBS_SIZE,
    LOW_LEVEL_ACTION_SPACE_SHAPE,
    MACRO_ACTION_SPACE_SHAPE,
    RAW_ACTION_SPACE_SHAPE,
    infer_boss_name,
    load_demonstrations,
)


DEFAULT_OBS_SIZE = int(os.environ.get("SILKSONGRL_BC_OBS_SIZE", str(LACE_COMPRESSED_DENSE_OBS_SIZE)))
DEFAULT_ACTION_SPACE = os.environ.get("SILKSONGRL_BC_ACTION_SPACE", "composite26")


def resolve_action_space_shape(name: str) -> list[int]:
    key = (name or "raw").strip().lower().replace("-", "_")
    if key in ("raw", "raw8", "raw_keys", "rawkeys", "raw_branches"):
        return RAW_ACTION_SPACE_SHAPE
    if key in ("composite", "composite26", "move_offense", "move_offense26"):
        return COMPOSITE_ACTION_SPACE_SHAPE
    if key in ("macro", "option", "option11", "macro11"):
        return MACRO_ACTION_SPACE_SHAPE
    if key in ("lowlevel", "low_level", "lowlevel30", "low_level30"):
        return LOW_LEVEL_ACTION_SPACE_SHAPE
    raise ValueError(f"Unknown action space '{name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior cloning pretraining from SilksongRL demos.")
    parser.add_argument("--demo-dir", default=os.path.join(DEFAULT_DEMO_ROOT, "lace_2"))
    parser.add_argument("--boss-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--obs-size", type=int, default=DEFAULT_OBS_SIZE)
    parser.add_argument(
        "--action-space",
        default=DEFAULT_ACTION_SPACE,
        choices=["raw", "composite26", "option11", "lowlevel30"],
        help="BC target action space. 'raw' trains move/look/jump/attack/dash/needle/bind/tool branches.",
    )
    parser.add_argument("--respect-masks", action="store_true")
    parser.add_argument("--branch-weights", default=None)
    parser.add_argument("--nonzero-boosts", default=None)
    parser.add_argument("--action-value-weights", default=None)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def set_learning_rate(model, learning_rate: float) -> None:
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate


def evaluate_bc(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int = 2048,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    exact_matches = []
    branch_matches = []
    branch_correct = []
    branch_total = []
    nonzero_correct = []
    nonzero_total = []

    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            obs_t = model._obs_to_tensor(observations[start:end])
            action_t = torch.as_tensor(actions[start:end], dtype=torch.long, device=model.device)
            dist = model.policy.get_distribution(obs_t)
            pred = dist.mode()
            exact_matches.append((pred == action_t).all(dim=1).float().cpu().numpy())
            branch_matches.append((pred == action_t).float().mean(dim=1).cpu().numpy())

            correct = (pred == action_t).detach().cpu().numpy()
            action_np = actions[start:end]
            branch_correct.append(correct.sum(axis=0))
            branch_total.append(np.full(actions.shape[1], len(action_np)))
            nonzero_mask = action_np != 0
            nonzero_correct.append((correct & nonzero_mask).sum(axis=0))
            nonzero_total.append(nonzero_mask.sum(axis=0))

    branch_correct_arr = np.stack(branch_correct).sum(axis=0)
    branch_total_arr = np.stack(branch_total).sum(axis=0)
    nonzero_correct_arr = np.stack(nonzero_correct).sum(axis=0)
    nonzero_total_arr = np.stack(nonzero_total).sum(axis=0)

    branch_acc = branch_correct_arr / np.maximum(branch_total_arr, 1)
    nonzero_recall = nonzero_correct_arr / np.maximum(nonzero_total_arr, 1)

    return (
        float(np.concatenate(exact_matches).mean()),
        float(np.concatenate(branch_matches).mean()),
        branch_acc,
        nonzero_recall,
    )


def remove_existing_model_dir(model_key: str) -> None:
    model_root = Path(rl_core.MODEL_ROOT).resolve()
    model_dir = (model_root / model_key).resolve()
    try:
        model_dir.relative_to(model_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete model directory outside MODEL_ROOT: {model_dir}") from exc

    if model_dir.is_dir():
        shutil.rmtree(model_dir)
        print(f"[BC] Removed existing model directory for fresh BC: {model_dir}")


def train_behavior_cloning(args: argparse.Namespace) -> None:
    boss_name = args.boss_name or infer_boss_name(args.demo_dir, min_steps=args.min_steps)
    action_space_shape = resolve_action_space_shape(args.action_space)

    dataset = load_demonstrations(
        demo_dir=args.demo_dir,
        boss_name=boss_name,
        obs_size=args.obs_size,
        action_shape=action_space_shape,
        min_steps=args.min_steps,
        max_samples=args.max_samples,
        drop_mask_violations=args.respect_masks,
    )

    obs_size = int(dataset.observations.shape[1])
    rl_core.initialize_model(
        obs_size,
        boss_name,
        action_space_shape,
        observation_type="vector",
        vector_obs_size=obs_size,
        is_eval=False,
    )
    model = rl_core.model
    if model is None:
        raise RuntimeError("Model failed to initialize")

    if args.fresh:
        remove_existing_model_dir(model.boss_name)
        rl_core.initialize_model(
            obs_size,
            boss_name,
            action_space_shape,
            observation_type="vector",
            vector_obs_size=obs_size,
            is_eval=False,
        )
        model = rl_core.model
        if model is None:
            raise RuntimeError("Model failed to reinitialize after --fresh")

    set_learning_rate(model, args.learning_rate)
    model.policy.set_training_mode(True)

    observations = dataset.observations
    observations = rl_core.augment_observation_batch(observations)
    actions = dataset.actions
    action_masks = dataset.action_masks if args.respect_masks else None
    branch_weights = parse_branch_weights(args.branch_weights, len(action_space_shape))
    nonzero_boosts = parse_nonzero_boosts(args.nonzero_boosts, len(action_space_shape))
    action_value_weights = parse_action_value_weights(args.action_value_weights, action_space_shape)

    print(f"[BC] Boss: {boss_name}")
    print(f"[BC] Action space: {args.action_space} {action_space_shape}")
    print(f"[BC] Demo files: {len(dataset.files)}, samples: {len(observations)}, obs_size: {obs_size}")
    print(f"[BC] Respect masks: {args.respect_masks}")
    print(
        f"[BC] Branch weights: {branch_weights}, nonzero boosts: {nonzero_boosts}, "
        f"action value weights: {action_value_weights}"
    )

    rng = np.random.default_rng()
    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(len(observations))
        losses = []

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            obs_t = model._obs_to_tensor(observations[batch_idx])
            action_t = torch.as_tensor(actions[batch_idx], dtype=torch.long, device=model.device)
            mask_batch = None
            if action_masks is not None:
                mask_batch = action_masks[batch_idx]

            dist = model.policy.get_distribution(obs_t, action_masks=mask_batch)
            loss = weighted_multidiscrete_bc_loss(
                dist,
                action_t,
                branch_weights=branch_weights,
                nonzero_boosts=nonzero_boosts,
                action_value_weights=action_value_weights,
            )

            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        exact_acc, branch_acc, per_branch_acc, nonzero_recall = evaluate_bc(model, observations, actions)
        print(
            f"[BC] Epoch {epoch}/{args.epochs}: loss={np.mean(losses):.4f}, "
            f"exact_acc={exact_acc * 100:.1f}%, branch_acc={branch_acc * 100:.1f}%"
        )
        print(
            "[BC]   per_branch_acc="
            + ",".join(f"{value * 100:.1f}" for value in per_branch_acc)
            + " nonzero_recall="
            + ",".join(f"{value * 100:.1f}" for value in nonzero_recall)
        )

    save_dir = model._boss_directory()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "checkpoint")
    model.save(save_path)
    print(f"[BC] Saved checkpoint: {save_path}.zip")


if __name__ == "__main__":
    train_behavior_cloning(parse_args())
