from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from demo_dataset import (
    BASE_OBS_SIZE,
    DEFAULT_DEMO_ROOT,
    LACE_COMPRESSED_DENSE_OBS_SIZE,
    LACE_COMPRESSED_SINGLE_FRAME_SIZE,
    LACE_COMPRESSED_FSM_ONE_HOT_SIZE,
    LACE_COMPRESSED_FSM_SEMANTIC_SIZE,
    LACE_DENSE_FRAME_STACK_SIZE,
    _adapt_observation_sequence,
    _extract_lace_single_frames,
    _load_file_records,
    _normalize,
)


PREDICTOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictors")
DEFAULT_HORIZONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
PER_HORIZON_FEATURES = 2 + 1 + 1 + LACE_COMPRESSED_FSM_SEMANTIC_SIZE
DEFAULT_FEATURE_DIM = len(DEFAULT_HORIZONS) * PER_HORIZON_FEATURES


class FuturePredictorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = DEFAULT_FEATURE_DIM, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FuturePredictorRuntime:
    def __init__(self, model: FuturePredictorNet, input_dim: int, feature_dim: int, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.device = device

    @classmethod
    def try_load(cls, boss_name: str, root: str = PREDICTOR_ROOT) -> Optional["FuturePredictorRuntime"]:
        boss_key = _normalize(boss_name)
        checkpoint_path = Path(root) / boss_key / "future_predictor.pt"
        if not checkpoint_path.exists():
            print(f"[FuturePredictor] No predictor found at {checkpoint_path}; PPO will run without predictor features")
            return None

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        input_dim = int(payload["input_dim"])
        feature_dim = int(payload["feature_dim"])
        model = FuturePredictorNet(input_dim=input_dim, output_dim=feature_dim)
        model.load_state_dict(payload["model_state_dict"])
        print(f"[FuturePredictor] Loaded predictor: {checkpoint_path}, input={input_dim}, features={feature_dim}")
        return cls(model, input_dim=input_dim, feature_dim=feature_dim)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[1] != self.input_dim:
            raise ValueError(f"Future predictor expected obs {self.input_dim}, got {obs.shape[1]}")

        with torch.no_grad():
            logits = self.model(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            features = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return features

    def augment_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.concatenate([obs, self.predict(obs)[0]], axis=0).astype(np.float32)

    def augment_observations(self, observations: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        features = []
        for start in range(0, len(obs), batch_size):
            features.append(self.predict(obs[start:start + batch_size]))
        return np.concatenate([obs, np.concatenate(features, axis=0)], axis=1).astype(np.float32)


def predictor_dir_for_boss(boss_name: str, root: str = PREDICTOR_ROOT) -> Path:
    return Path(root) / _normalize(boss_name)


def build_future_predictor_dataset(
    demo_dir: Union[str, Sequence[str]] = DEFAULT_DEMO_ROOT,
    boss_name: str = "Lace Boss2 New",
    obs_size: int = LACE_COMPRESSED_DENSE_OBS_SIZE,
    horizons: Sequence[float] = DEFAULT_HORIZONS,
    min_steps: int = 20,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    target_boss = _normalize(boss_name)
    observations: List[List[float]] = []
    labels: List[List[float]] = []

    for path in _iter_jsonl_files(demo_dir):
        records = _load_file_records(path)
        if len(records) < min_steps:
            continue

        file_boss = records[0].get("boss_name", "")
        if _normalize(file_boss) != target_boss:
            continue

        adapted = _adapt_observation_sequence(records, obs_size)
        single_frames = _extract_lace_single_frames(records)
        times = _record_times(records)
        if not adapted or not single_frames:
            continue

        for index, (obs, frame) in enumerate(zip(adapted, single_frames)):
            if not isinstance(obs, list) or frame is None:
                continue

            label = _build_label_for_index(index, single_frames, times, horizons)
            if label is None:
                continue

            observations.append(obs)
            labels.append(label)
            if max_samples is not None and len(observations) >= max_samples:
                return (
                    np.asarray(observations, dtype=np.float32),
                    np.asarray(labels, dtype=np.float32),
                )

    if not observations:
        raise ValueError(f"No predictor samples found in {demo_dir} for boss '{boss_name}'")

    return np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _iter_jsonl_files(demo_dir: Union[str, Sequence[str]]) -> Iterable[Path]:
    roots = [demo_dir] if isinstance(demo_dir, (str, os.PathLike)) else list(demo_dir)
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        yield from sorted(root.rglob("*.jsonl"))


def _record_times(records: Sequence[dict]) -> List[float]:
    times: List[float] = []
    fallback = 0.0
    for record in records:
        try:
            fallback = float(record.get("time"))
        except (TypeError, ValueError):
            fallback += 0.05
        times.append(fallback)
    return times


def _build_label_for_index(
    index: int,
    frames: Sequence[Optional[List[float]]],
    times: Sequence[float],
    horizons: Sequence[float],
) -> Optional[List[float]]:
    current_frame = frames[index]
    if current_frame is None:
        return None

    label: List[float] = []
    for horizon in horizons:
        future_index = _future_index(index, times, horizon)
        if future_index is None or future_index >= len(frames) or frames[future_index] is None:
            return None
        label.extend(_future_features(current_frame, frames[future_index]))

    return label


def _future_index(index: int, times: Sequence[float], horizon: float) -> Optional[int]:
    target = times[index] + float(horizon)
    for candidate in range(index + 1, len(times)):
        if times[candidate] >= target:
            return candidate
    return None


def _future_features(current_frame: List[float], future_frame: List[float]) -> List[float]:
    rel_x, rel_y = _future_relative_position(current_frame, future_frame)
    semantic = _future_semantic_flags(future_frame)
    danger = semantic[0] if semantic else 0.0
    can_punish = semantic[1] if len(semantic) > 1 else 0.0
    return [rel_x, rel_y, danger, can_punish] + semantic


def _future_relative_position(current_frame: List[float], future_frame: List[float]) -> Tuple[float, float]:
    arena_width = 44.0
    arena_height = 14.0
    hero_x = 33.0 + current_frame[0] * arena_width
    hero_y = 96.0 + current_frame[1] * arena_height
    boss_x = 33.0 + future_frame[5] * arena_width
    boss_y = 96.0 + future_frame[6] * arena_height
    rel_x = boss_x - hero_x
    rel_y = boss_y - hero_y
    return (
        float(np.clip(0.5 + rel_x / (2.0 * arena_width), 0.0, 1.0)),
        float(np.clip(0.5 + rel_y / (2.0 * arena_height), 0.0, 1.0)),
    )


def _future_semantic_flags(frame: List[float]) -> List[float]:
    base = BASE_OBS_SIZE
    start = base + LACE_COMPRESSED_FSM_ONE_HOT_SIZE
    end = start + LACE_COMPRESSED_FSM_SEMANTIC_SIZE
    if len(frame) < end:
        return [0.0] * LACE_COMPRESSED_FSM_SEMANTIC_SIZE
    return [float(np.clip(value, 0.0, 1.0)) for value in frame[start:end]]


def split_train_val(
    observations: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(observations))
    val_size = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return observations[train_idx], labels[train_idx], observations[val_idx], labels[val_idx]
