from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from demo_dataset import DEFAULT_DEMO_ROOT, LACE_COMPRESSED_DENSE_OBS_SIZE
from future_predictor import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_HORIZONS,
    FuturePredictorNet,
    build_future_predictor_dataset,
    predictor_dir_for_boss,
    split_train_val,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future state predictor from SilksongRL demos.")
    parser.add_argument("--demo-dir", default=os.path.join(DEFAULT_DEMO_ROOT, "lace_2"))
    parser.add_argument(
        "--extra-demo-dir",
        action="append",
        default=[],
        help="Additional JSONL roots, e.g. python-client/agent_traces/lace_boss2_new. Can be passed multiple times.",
    )
    parser.add_argument("--boss-name", default="Lace Boss2 New")
    parser.add_argument("--obs-size", type=int, default=LACE_COMPRESSED_DENSE_OBS_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def predictor_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    per_horizon = 20
    losses = []
    for start in range(0, labels.shape[1], per_horizon):
        losses.append(F.mse_loss(pred[:, start:start + 2], labels[:, start:start + 2]))
        losses.append(F.binary_cross_entropy_with_logits(logits[:, start + 2:start + per_horizon], labels[:, start + 2:start + per_horizon]))
    return torch.stack(losses).mean()


def evaluate(model: FuturePredictorNet, observations: np.ndarray, labels: np.ndarray, batch_size: int = 2048) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(observations[start:start + batch_size], dtype=torch.float32)
            preds.append(torch.sigmoid(model(obs)).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    labels = np.asarray(labels, dtype=np.float32)
    per_horizon = 20
    metrics = {}
    dxdy_mae = []
    danger_acc = []
    punish_acc = []
    semantic_acc = []
    for horizon_index, horizon in enumerate(DEFAULT_HORIZONS):
        start = horizon_index * per_horizon
        dxdy_mae.append(float(np.mean(np.abs(pred[:, start:start + 2] - labels[:, start:start + 2]))))
        danger_acc.append(float(np.mean((pred[:, start + 2] >= 0.5) == (labels[:, start + 2] >= 0.5))))
        punish_acc.append(float(np.mean((pred[:, start + 3] >= 0.5) == (labels[:, start + 3] >= 0.5))))
        semantic_acc.append(float(np.mean((pred[:, start + 4:start + per_horizon] >= 0.5) == (labels[:, start + 4:start + per_horizon] >= 0.5))))
        metrics[f"{horizon:.1f}s_dxdy_mae"] = dxdy_mae[-1]
        metrics[f"{horizon:.1f}s_danger_acc"] = danger_acc[-1]
        metrics[f"{horizon:.1f}s_canpunish_acc"] = punish_acc[-1]
        metrics[f"{horizon:.1f}s_semantic_acc"] = semantic_acc[-1]

    metrics["mean_dxdy_mae"] = float(np.mean(dxdy_mae))
    metrics["mean_danger_acc"] = float(np.mean(danger_acc))
    metrics["mean_canpunish_acc"] = float(np.mean(punish_acc))
    metrics["mean_semantic_acc"] = float(np.mean(semantic_acc))
    return metrics


def main() -> None:
    args = parse_args()
    demo_dirs = [args.demo_dir] + list(args.extra_demo_dir or [])
    observations, labels = build_future_predictor_dataset(
        demo_dir=demo_dirs,
        boss_name=args.boss_name,
        obs_size=args.obs_size,
        min_steps=args.min_steps,
        max_samples=args.max_samples,
    )
    train_obs, train_labels, val_obs, val_labels = split_train_val(observations, labels, args.val_fraction)
    model = FuturePredictorNet(input_dim=train_obs.shape[1], output_dim=train_labels.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng()

    print(
        f"[FuturePredictor] demo_dirs={demo_dirs}, samples train={len(train_obs)}, "
        f"val={len(val_obs)}, obs={train_obs.shape[1]}, labels={train_labels.shape[1]}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = rng.permutation(len(train_obs))
        losses = []
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            obs_t = torch.as_tensor(train_obs[batch], dtype=torch.float32)
            label_t = torch.as_tensor(train_labels[batch], dtype=torch.float32)
            loss = predictor_loss(model(obs_t), label_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        metrics = evaluate(model, val_obs, val_labels)
        print(
            f"[FuturePredictor] epoch {epoch}/{args.epochs}: loss={np.mean(losses):.4f}, "
            f"dxdy_mae={metrics['mean_dxdy_mae']:.4f}, danger_acc={metrics['mean_danger_acc']*100:.1f}%, "
            f"canpunish_acc={metrics['mean_canpunish_acc']*100:.1f}%, semantic_acc={metrics['mean_semantic_acc']*100:.1f}%"
        )

    save_dir = predictor_dir_for_boss(args.boss_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_obs.shape[1]),
            "feature_dim": int(train_labels.shape[1]),
            "horizons": list(DEFAULT_HORIZONS),
            "demo_dirs": demo_dirs,
            "metrics": evaluate(model, val_obs, val_labels),
        },
        save_dir / "future_predictor.pt",
    )
    with open(save_dir / "future_predictor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluate(model, val_obs, val_labels), f, indent=2)
    print(f"[FuturePredictor] Saved: {save_dir / 'future_predictor.pt'}")


if __name__ == "__main__":
    main()
