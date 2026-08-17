from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from demo_dataset import (
    BASE_OBS_SIZE,
    DEFAULT_DEMO_ROOT,
    LACE_COMPRESSED_DENSE_OBS_SIZE,
    LACE_COMPRESSED_FSM_SIZE,
    RAW_ACTION_SPACE_SHAPE,
    _adapt_action,
    _adapt_mask,
    _adapt_observation_sequence,
    _load_file_records,
    _normalize,
)
from frame_encoder import FrameStackFeatureExtractor
from future_predictor import FuturePredictorRuntime


MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_models")
AGENT_TRACE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_traces")
WIN_RATE_WINDOW = 25
FRAME_STACK_SIZE = int(os.environ.get("SILKSONGRL_FRAME_STACK", "16"))
BOSS_HP_OBS_INDEX = 9
BOSS_PHASE_OBS_OFFSET = BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE
BOSS_PHASE_OBS_SIZE = 3

DQN_BUFFER_SIZE = int(os.environ.get("SILKSONGRL_DQN_BUFFER_SIZE", "100000"))
DQN_BATCH_SIZE = int(os.environ.get("SILKSONGRL_DQN_BATCH_SIZE", "256"))
DQN_ELITE_BUFFER_SIZE = int(os.environ.get("SILKSONGRL_DQN_ELITE_BUFFER_SIZE", "30000"))
DQN_ELITE_BATCH_FRACTION = float(os.environ.get("SILKSONGRL_DQN_ELITE_BATCH_FRACTION", "0.25"))
DQN_ELITE_MIN_PHASE = float(os.environ.get("SILKSONGRL_DQN_ELITE_MIN_PHASE", "2"))
DQN_ELITE_MAX_BOSS_HP_PERCENT = float(os.environ.get("SILKSONGRL_DQN_ELITE_MAX_BOSS_HP_PERCENT", "50"))
DQN_GAMMA = float(os.environ.get("SILKSONGRL_DQN_GAMMA", "0.997"))
DQN_N_STEP = int(os.environ.get("SILKSONGRL_DQN_N_STEP", "5"))
DQN_LR = float(os.environ.get("SILKSONGRL_DQN_LEARNING_RATE", "1e-4"))
DQN_LEARNING_STARTS = int(os.environ.get("SILKSONGRL_DQN_LEARNING_STARTS", "5000"))
DQN_TRAIN_FREQ = int(os.environ.get("SILKSONGRL_DQN_TRAIN_FREQ", "1"))
DQN_GRADIENT_STEPS = int(os.environ.get("SILKSONGRL_DQN_GRADIENT_STEPS", "1"))
DQN_TARGET_UPDATE_INTERVAL = int(os.environ.get("SILKSONGRL_DQN_TARGET_UPDATE_INTERVAL", "1500"))
DQN_SAVE_FREQ_EPISODES = int(os.environ.get("SILKSONGRL_DQN_SAVE_FREQ_EPISODES", "50"))
DQN_EPS_START = float(os.environ.get("SILKSONGRL_DQN_EPS_START", "0.20"))
DQN_EPS_FINAL = float(os.environ.get("SILKSONGRL_DQN_EPS_FINAL", "0.05"))
DQN_EPS_DECAY_STEPS = int(os.environ.get("SILKSONGRL_DQN_EPS_DECAY_STEPS", "300000"))
DQN_PER_ALPHA = float(os.environ.get("SILKSONGRL_DQN_PER_ALPHA", "0.6"))
DQN_PER_BETA_START = float(os.environ.get("SILKSONGRL_DQN_PER_BETA_START", "0.4"))
DQN_PER_BETA_STEPS = int(os.environ.get("SILKSONGRL_DQN_PER_BETA_STEPS", "200000"))
DQN_PER_EPS = float(os.environ.get("SILKSONGRL_DQN_PER_EPS", "1e-4"))
DQN_PREFILL_ENABLED = os.environ.get("SILKSONGRL_DQN_PREFILL", "1") != "0"
DQN_PREFILL_MAX_SAMPLES = int(os.environ.get("SILKSONGRL_DQN_PREFILL_MAX_SAMPLES", "5000"))
DQN_PREFILL_INCLUDE_AGENT_TRACES = os.environ.get("SILKSONGRL_DQN_PREFILL_INCLUDE_AGENT_TRACES", "0") == "1"
DQN_PREFILL_MAX_FILES = int(os.environ.get("SILKSONGRL_DQN_PREFILL_MAX_FILES", "200"))
DQN_REWARD_NORM_ENABLED = os.environ.get("SILKSONGRL_DQN_REWARD_NORM", "1") != "0"
DQN_REWARD_NORM_CLIP = float(os.environ.get("SILKSONGRL_DQN_REWARD_NORM_CLIP", "10"))
PREDICTOR_ENABLED = os.environ.get("SILKSONGRL_FUTURE_PREDICTOR", "1") != "0"
AGENT_TRACE_ENABLED = os.environ.get("SILKSONGRL_SAVE_AGENT_TRACES", "1") != "0"
AGENT_TRACE_STEP_SECONDS = float(os.environ.get("SILKSONGRL_AGENT_TRACE_STEP_SECONDS", "0.05"))
AGENT_TRACE_MIN_STEPS = int(os.environ.get("SILKSONGRL_AGENT_TRACE_MIN_STEPS", "20"))


model: Optional["DQNAgent"] = None
unity_obs_dim: Optional[int] = None
obs_dim: Optional[int] = None
action_shape: Optional[List[int]] = None
current_boss: Optional[str] = None
eval_mode: bool = False
future_predictor: Optional[FuturePredictorRuntime] = None
reward_normalizer: Optional["RunningRewardNormalizer"] = None
agent_trace_recorder: Optional["AgentTraceRecorder"] = None


def normalize_boss_name(boss_name: str) -> str:
    return boss_name.replace(" ", "_").lower()


def model_key_for_action_space(
    boss_name: str,
    action_space_shape: Optional[List[int]],
    observation_size: Optional[int] = None,
    predictor_feature_dim: int = 0,
) -> str:
    key = normalize_boss_name(boss_name)
    suffix = "_dqn"
    if list(action_space_shape or []) == [30]:
        suffix += "_lowlevel30_raycast32_hitbox12"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        suffix += "_dueling_per_nstep_frameenc"
    elif list(action_space_shape or []) == [26]:
        suffix += "_move_offense26"
        if observation_size in (2144, 2192):
            suffix += "_fsmsemantic80_facing"
        if observation_size == 2192:
            suffix += "_sparsephase"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        suffix += "_dueling_per_nstep_frameenc"
    elif list(action_space_shape or []) == [11]:
        suffix += "_option11"
        if observation_size in (2144, 2192):
            suffix += "_fsmsemantic80_facing"
        if observation_size == 2192:
            suffix += "_sparsephase"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        suffix += "_dueling_per_nstep_frameenc"
    elif list(action_space_shape or []) == RAW_ACTION_SPACE_SHAPE:
        suffix += "_rawbranch8_joint1152_raycast32_hitbox12"
        if observation_size in (2144, 2192):
            suffix += "_fsmsemantic80_facing"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        suffix += "_dueling_per_nstep_frameenc"
    return f"{key}{suffix}"


class RunningRewardNormalizer:
    def __init__(self, epsilon: float = 1e-4) -> None:
        self.count = float(epsilon)
        self.mean = 0.0
        self.m2 = 1.0 * float(epsilon)

    @property
    def std(self) -> float:
        variance = max(self.m2 / max(self.count, 1.0), 1e-6)
        return float(np.sqrt(variance))

    def normalize(self, value: float) -> float:
        value = float(value)
        delta = value - self.mean
        self.count += 1.0
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        return float(np.clip((value - self.mean) / self.std, -DQN_REWARD_NORM_CLIP, DQN_REWARD_NORM_CLIP))

    def state_dict(self) -> Dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, data: Dict[str, Any]) -> None:
        self.count = float(data.get("count", self.count))
        self.mean = float(data.get("mean", self.mean))
        self.m2 = float(data.get("m2", self.m2))


class AgentTraceRecorder:
    def __init__(self, root: str, step_seconds: float = 0.05, min_steps: int = 20) -> None:
        self.root = root
        self.step_seconds = max(0.001, float(step_seconds))
        self.min_steps = max(1, int(min_steps))
        self.boss_name: Optional[str] = None
        self.boss_key: Optional[str] = None
        self.action_space_shape: Optional[List[int]] = None
        self.episode_id: Optional[str] = None
        self.path: Optional[str] = None
        self.file = None
        self.step = 0

    def start_boss(self, boss_name: str, action_space_shape: Optional[List[int]]) -> None:
        self.close("boss_changed")
        self.boss_name = boss_name
        self.boss_key = _normalize(boss_name)
        self.action_space_shape = list(action_space_shape or [])

    def record(
        self,
        state: Sequence[float],
        action: Sequence[int],
        reward_raw: float,
        reward_normalized: float,
        done: bool,
        action_mask: Optional[Sequence[int]] = None,
        reward_components: Optional[Any] = None,
        teacher_action: Optional[Sequence[int]] = None,
    ) -> None:
        if not self.boss_name or not self.boss_key:
            return
        self._ensure_open()
        if self.file is None:
            return

        record = {
            "version": 8,
            "source": "dqn_agent",
            "action_space_shape": self.action_space_shape,
            "boss_key": self.boss_key,
            "boss_name": self.boss_name,
            "episode_id": self.episode_id,
            "step": self.step,
            "time": self.step * self.step_seconds,
            "done": bool(done),
            "terminal_reason": "done" if done else "",
            "observation_size": len(state),
            "observation": [float(v) for v in state],
            "action": [int(v) for v in action],
            "action_mask": [int(v) for v in action_mask] if action_mask is not None else None,
            "teacher_action": [int(v) for v in teacher_action] if teacher_action is not None else None,
            "reward_raw": float(reward_raw),
            "reward_normalized": float(reward_normalized),
            "reward_components": _jsonable(reward_components),
        }
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.step += 1
        if done:
            self.close("done")

    def close(self, reason: str = "closed") -> None:
        if self.file is None:
            return
        path = self.path
        steps = self.step
        self.file.close()
        self.file = None
        self.path = None
        self.episode_id = None
        self.step = 0
        if path and steps < self.min_steps:
            try:
                os.remove(path)
            except OSError:
                pass
        elif path:
            print(f"[AgentTrace] Saved {path} ({reason}), steps={steps}")

    def _ensure_open(self) -> None:
        if self.file is not None:
            return
        assert self.boss_key is not None
        folder = os.path.join(self.root, self.boss_key)
        os.makedirs(folder, exist_ok=True)
        millis = int((time.time() % 1.0) * 1000)
        self.episode_id = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{millis:03d}"
        self.path = os.path.join(folder, f"{self.episode_id}.jsonl")
        self.file = open(self.path, "w", encoding="utf-8")
        self.step = 0


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_dim: int,
        stacked_obs_dim: int,
        predictor_feature_dim: int,
    ) -> None:
        super().__init__()
        observation_space = spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)
        self.features = FrameStackFeatureExtractor(
            observation_space,
            features_dim=512,
            frame_stack=FRAME_STACK_SIZE,
            per_frame_features=64,
            stacked_obs_dim=stacked_obs_dim,
            extra_features_dim=predictor_feature_dim,
        )
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features(obs)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, action_dim: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.obs_size = int(obs_size)
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        self.obs = np.zeros((self.capacity, self.obs_size), dtype=np.float16)
        self.next_obs = np.zeros((self.capacity, self.obs_size), dtype=np.float16)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.action_masks = np.ones((self.capacity, self.action_dim), dtype=np.bool_)
        self.next_action_masks = np.ones((self.capacity, self.action_dim), dtype=np.bool_)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
        discount: float,
        priority: Optional[float] = None,
    ) -> None:
        index = self.pos
        self.obs[index] = obs.astype(np.float16, copy=False)
        self.next_obs[index] = next_obs.astype(np.float16, copy=False)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.discounts[index] = float(discount)
        self.dones[index] = bool(done)
        self.action_masks[index] = action_mask.astype(np.bool_, copy=False)
        self.next_action_masks[index] = next_action_mask.astype(np.bool_, copy=False)
        priority_value = float(priority) if priority is not None else self.max_priority
        priority_value = max(priority_value, DQN_PER_EPS)
        self.priorities[index] = priority_value
        self.max_priority = max(self.max_priority, priority_value)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise ValueError("Cannot sample from empty replay buffer")
        priorities = self.priorities[:self.size]
        scaled = np.power(np.maximum(priorities, DQN_PER_EPS), self.alpha)
        probs = scaled / scaled.sum()
        indices = np.random.choice(self.size, size=batch_size, replace=self.size < batch_size, p=probs)
        weights = np.power(self.size * probs[indices], -float(beta))
        weights = weights / weights.max()
        return {
            "indices": indices,
            "obs": self.obs[indices].astype(np.float32),
            "next_obs": self.next_obs[indices].astype(np.float32),
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "discounts": self.discounts[indices],
            "dones": self.dones[indices],
            "action_masks": self.action_masks[indices],
            "next_action_masks": self.next_action_masks[indices],
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.maximum(np.asarray(priorities, dtype=np.float32), DQN_PER_EPS)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))


@dataclass
class PendingTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    priority: float


@dataclass
class ReplayTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    discount: float
    priority: float


def add_replay_transition(buffer: PrioritizedReplayBuffer, transition: ReplayTransition) -> None:
    buffer.add(
        transition.obs,
        transition.action,
        transition.reward,
        transition.next_obs,
        transition.done,
        transition.action_mask,
        transition.next_action_mask,
        transition.discount,
        priority=transition.priority,
    )


class DQNAgent:
    def __init__(
        self,
        boss_name: str,
        model_name: str,
        obs_size: int,
        unity_obs_size: int,
        action_dim: int,
        action_shape: Sequence[int],
        predictor_feature_dim: int,
        eval_mode: bool = False,
    ) -> None:
        self.boss_name = boss_name
        self.model_name = model_name
        self.obs_size = int(obs_size)
        self.unity_obs_size = int(unity_obs_size)
        self.action_dim = int(action_dim)
        self.action_shape = list(action_shape or [action_dim])
        self.predictor_feature_dim = int(predictor_feature_dim)
        self.eval_mode = bool(eval_mode)
        self.device = torch.device("cpu")
        self.online = DuelingQNetwork(obs_size, action_dim, unity_obs_size, predictor_feature_dim).to(self.device)
        self.target = DuelingQNetwork(obs_size, action_dim, unity_obs_size, predictor_feature_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=DQN_LR, weight_decay=1e-5)
        self.replay = PrioritizedReplayBuffer(DQN_BUFFER_SIZE, obs_size, action_dim, DQN_PER_ALPHA)
        self.elite_replay = PrioritizedReplayBuffer(DQN_ELITE_BUFFER_SIZE, obs_size, action_dim, DQN_PER_ALPHA)
        self.pending: Deque[PendingTransition] = deque()
        self.current_episode_replay: List[ReplayTransition] = []
        self.total_env_steps = 0
        self.total_train_steps = 0
        self.episodes_completed = 0
        self.episode_reward = 0.0
        self.episode_rewards: List[float] = []
        self.episode_boss_hp: List[float] = []
        self.episode_phases: List[float] = []
        self.episode_action_counts: List[List[int]] = []
        self.current_action_counts = [0 for _ in range(action_dim)]
        self.best_score = float("-inf")
        self.best_avg_boss_hp = float("inf")
        self.best_avg_reward = float("-inf")
        self.best_avg_phase = 1.0
        self.last_loss = float("nan")
        self.last_td_abs = float("nan")

    @property
    def model_dir(self) -> str:
        return os.path.join(MODEL_ROOT, self.model_name)

    def epsilon(self) -> float:
        if self.eval_mode:
            return 0.0
        progress = min(1.0, self.total_env_steps / max(1, DQN_EPS_DECAY_STEPS))
        return float(DQN_EPS_START + progress * (DQN_EPS_FINAL - DQN_EPS_START))

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        mask = ensure_action_mask(action_mask, self.action_dim, self.action_shape)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            legal = np.array([0], dtype=np.int64)
        if not self.eval_mode and random.random() < self.epsilon():
            return int(np.random.choice(legal))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=self.device)
            q = self.online(obs_t).cpu().numpy()[0]
        q[~mask] = -1e9
        return int(np.argmax(q))

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.total_env_steps += 1
        self.episode_reward += float(reward)
        if 0 <= action < len(self.current_action_counts):
            self.current_action_counts[action] += 1

        priority = abs(float(reward)) + 1.0
        self.pending.append(
            PendingTransition(
                obs=obs,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                done=bool(done),
                action_mask=action_mask,
                next_action_mask=next_action_mask,
                priority=priority,
            )
        )
        if len(self.pending) >= DQN_N_STEP:
            self._flush_one_pending()
        if done:
            while self.pending:
                self._flush_one_pending()
            self._finish_episode(next_obs)

        can_train = self.replay.size >= max(DQN_BATCH_SIZE, DQN_LEARNING_STARTS)
        if not self.eval_mode and can_train and self.total_env_steps % DQN_TRAIN_FREQ == 0:
            for _ in range(DQN_GRADIENT_STEPS):
                self.train_step()

        if self.total_env_steps > 0 and self.total_env_steps % DQN_TARGET_UPDATE_INTERVAL == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _flush_one_pending(self) -> None:
        reward = 0.0
        discount = 1.0
        next_obs = self.pending[0].next_obs
        next_mask = self.pending[0].next_action_mask
        done = False
        priority = 1.0
        for index, transition in enumerate(list(self.pending)[:DQN_N_STEP]):
            reward += discount * transition.reward
            priority += transition.priority
            next_obs = transition.next_obs
            next_mask = transition.next_action_mask
            done = transition.done
            if done:
                discount = 0.0
                break
            discount *= DQN_GAMMA
        first = self.pending.popleft()
        replay_transition = ReplayTransition(
            obs=first.obs,
            action=first.action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            action_mask=first.action_mask,
            next_action_mask=next_mask,
            discount=discount,
            priority=priority / max(1, min(DQN_N_STEP, len(self.pending) + 1)),
        )
        add_replay_transition(self.replay, replay_transition)
        self.current_episode_replay.append(replay_transition)

    def train_step(self) -> None:
        if self.replay.size < max(DQN_BATCH_SIZE, DQN_LEARNING_STARTS):
            return
        beta_progress = min(1.0, self.total_train_steps / max(1, DQN_PER_BETA_STEPS))
        beta = DQN_PER_BETA_START + beta_progress * (1.0 - DQN_PER_BETA_START)
        batch, main_count, elite_count = self._sample_training_batch(beta)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(batch["discounts"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_action_masks"], dtype=torch.bool, device=self.device)

        q_values = self.online(obs).gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_online_q = self.online(next_obs)
            next_online_q = next_online_q.masked_fill(~next_masks, -1e9)
            next_actions = next_online_q.argmax(dim=1)
            next_target_q = self.target(next_obs).gather(1, next_actions.view(-1, 1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * discounts * next_target_q
        td_error = target_q - q_values
        loss = (F.smooth_l1_loss(q_values, target_q, reduction="none") * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        priorities = td_error.detach().abs().cpu().numpy() + DQN_PER_EPS
        self.replay.update_priorities(batch["main_indices"], priorities[:main_count])
        if elite_count > 0:
            self.elite_replay.update_priorities(batch["elite_indices"], priorities[main_count:])
        self.total_train_steps += 1
        self.last_loss = float(loss.detach().cpu().item())
        self.last_td_abs = float(np.mean(priorities))

    def _sample_training_batch(self, beta: float) -> Tuple[Dict[str, np.ndarray], int, int]:
        elite_count = 0
        if self.elite_replay.size > 0 and DQN_ELITE_BATCH_FRACTION > 0:
            elite_count = int(round(DQN_BATCH_SIZE * DQN_ELITE_BATCH_FRACTION))
            elite_count = min(max(1, elite_count), DQN_BATCH_SIZE - 1)
        main_count = DQN_BATCH_SIZE - elite_count
        main_batch = self.replay.sample(main_count, beta)
        if elite_count <= 0:
            main_batch["main_indices"] = main_batch.pop("indices")
            main_batch["elite_indices"] = np.asarray([], dtype=np.int64)
            return main_batch, main_count, 0

        elite_batch = self.elite_replay.sample(elite_count, beta)
        batch: Dict[str, np.ndarray] = {
            key: np.concatenate([main_batch[key], elite_batch[key]], axis=0)
            for key in (
                "obs",
                "next_obs",
                "actions",
                "rewards",
                "discounts",
                "dones",
                "action_masks",
                "next_action_masks",
                "weights",
            )
        }
        batch["main_indices"] = main_batch["indices"]
        batch["elite_indices"] = elite_batch["indices"]
        return batch, main_count, elite_count

    def _finish_episode(self, terminal_obs: np.ndarray) -> None:
        boss_hp = extract_boss_hp_percent(terminal_obs)
        phase = extract_phase(terminal_obs)
        if self._is_elite_episode(boss_hp, phase):
            for transition in self.current_episode_replay:
                add_replay_transition(self.elite_replay, transition)
            print(
                f"[DQN] Elite episode stored: {len(self.current_episode_replay)} transitions, "
                f"elite replay {self.elite_replay.size}"
            )
        self.episodes_completed += 1
        self.episode_rewards.append(float(self.episode_reward))
        self.episode_boss_hp.append(float(boss_hp))
        self.episode_phases.append(float(phase))
        self.episode_action_counts.append(list(self.current_action_counts))
        print(
            f"[DQN] Episode {self.episodes_completed} reward: {self.episode_reward:.2f}, "
            f"boss remaining HP: {boss_hp:.1f}%, phase: {phase:.1f}, "
            f"epsilon {self.epsilon():.3f}, replay {self.replay.size}, elite {self.elite_replay.size}, "
            f"loss {self.last_loss:.4f}, td_abs {self.last_td_abs:.4f}"
        )
        self.episode_reward = 0.0
        self.current_action_counts = [0 for _ in range(self.action_dim)]
        self.current_episode_replay = []

        if self.episodes_completed % WIN_RATE_WINDOW == 0:
            self._print_window_stats()
        if DQN_SAVE_FREQ_EPISODES and self.episodes_completed % DQN_SAVE_FREQ_EPISODES == 0:
            self.save("checkpoint")

    def _is_elite_episode(self, boss_hp: float, phase: float) -> bool:
        return (
            np.isfinite(phase) and phase >= DQN_ELITE_MIN_PHASE
        ) or (
            np.isfinite(boss_hp) and boss_hp < DQN_ELITE_MAX_BOSS_HP_PERCENT
        )

    def _print_window_stats(self) -> None:
        rewards = self.episode_rewards[-WIN_RATE_WINDOW:]
        hp_values = [value for value in self.episode_boss_hp[-WIN_RATE_WINDOW:] if not np.isnan(value)]
        phases = [value for value in self.episode_phases[-WIN_RATE_WINDOW:] if not np.isnan(value)]
        action_counts = self.episode_action_counts[-WIN_RATE_WINDOW:]
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_hp = float(np.mean(hp_values)) if hp_values else float("nan")
        avg_phase = float(np.mean(phases)) if phases else 1.0
        score = best_score(avg_hp, avg_reward, avg_phase)
        avg_action_counts = np.asarray(action_counts, dtype=np.float32).mean(axis=0).tolist() if action_counts else []
        print(
            f"[DQN] Recent {len(rewards)} episode stats after {self.episodes_completed} episodes: "
            f"avg reward {avg_reward:.2f}, avg boss remaining HP {avg_hp:.1f}%, "
            f"avg phase {avg_phase:.2f}, score {score:.2f}, replay {self.replay.size}, "
            f"elite {self.elite_replay.size}"
        )
        print(f"[DQN] Recent action distribution: {format_joint_action_distribution(avg_action_counts, self.action_shape)}")
        if score > self.best_score:
            improvement = score - self.best_score if np.isfinite(self.best_score) else 0.0
            self.best_score = score
            self.best_avg_boss_hp = avg_hp
            self.best_avg_reward = avg_reward
            self.best_avg_phase = avg_phase
            self.save("best_checkpoint")
            print(
                f"[DQN] Best checkpoint saved: score {score:.2f} "
                f"(+{improvement:.2f}), avg HP {avg_hp:.1f}%, avg phase {avg_phase:.2f}"
            )

    def save(self, name: str) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{name}.pt")
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "boss_name": self.boss_name,
                "model_name": self.model_name,
                "obs_size": self.obs_size,
                "unity_obs_size": self.unity_obs_size,
                "action_dim": self.action_dim,
                "action_shape": self.action_shape,
                "predictor_feature_dim": self.predictor_feature_dim,
                "total_env_steps": self.total_env_steps,
                "total_train_steps": self.total_train_steps,
                "episodes_completed": self.episodes_completed,
                "episode_rewards": self.episode_rewards,
                "episode_boss_hp": self.episode_boss_hp,
                "episode_phases": self.episode_phases,
                "episode_action_counts": self.episode_action_counts,
                "best_score": self.best_score,
                "best_avg_boss_hp": self.best_avg_boss_hp,
                "best_avg_reward": self.best_avg_reward,
                "best_avg_phase": self.best_avg_phase,
            },
            path,
        )
        print(f"[DQN] Saved checkpoint: {path}")

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload.get("target", payload["online"]))
        try:
            self.optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            pass
        self.total_env_steps = int(payload.get("total_env_steps", 0))
        self.total_train_steps = int(payload.get("total_train_steps", 0))
        self.episodes_completed = int(payload.get("episodes_completed", 0))
        self.episode_rewards = list(payload.get("episode_rewards", []))
        self.episode_boss_hp = list(payload.get("episode_boss_hp", []))
        self.episode_phases = list(payload.get("episode_phases", []))
        self.episode_action_counts = list(payload.get("episode_action_counts", []))
        self.best_score = float(payload.get("best_score", float("-inf")))
        self.best_avg_boss_hp = float(payload.get("best_avg_boss_hp", float("inf")))
        self.best_avg_reward = float(payload.get("best_avg_reward", float("-inf")))
        self.best_avg_phase = float(payload.get("best_avg_phase", 1.0))


def best_score(avg_boss_hp: float, avg_reward: float, avg_phase: float) -> float:
    if np.isnan(avg_boss_hp) or np.isnan(avg_reward) or np.isnan(avg_phase):
        return float("-inf")
    return float(avg_reward) - float(avg_boss_hp) + float(avg_phase) * 2.0


def action_dim_from_shape(shape: Optional[Sequence[int]]) -> int:
    dim = 1
    for value in list(shape or [1]):
        dim *= max(1, int(value))
    return int(dim)


def flatten_action(action: Sequence[int], shape: Sequence[int]) -> int:
    values = list(action or [])
    shape_values = list(shape or [])
    if len(shape_values) == 1:
        return int(values[0]) if values else 0
    index = 0
    stride = 1
    for branch_value, branch_dim in zip(reversed(values), reversed(shape_values)):
        value = int(np.clip(int(branch_value), 0, int(branch_dim) - 1))
        index += value * stride
        stride *= int(branch_dim)
    return int(np.clip(index, 0, action_dim_from_shape(shape_values) - 1))


def unflatten_action(index: int, shape: Sequence[int]) -> List[int]:
    shape_values = list(shape or [])
    if len(shape_values) == 1:
        return [int(np.clip(index, 0, shape_values[0] - 1))]
    value = int(np.clip(index, 0, action_dim_from_shape(shape_values) - 1))
    action = [0 for _ in shape_values]
    for branch in range(len(shape_values) - 1, -1, -1):
        action[branch] = value % int(shape_values[branch])
        value //= int(shape_values[branch])
    return action


def branch_mask_to_joint_mask(mask: Sequence[Any], shape: Sequence[int]) -> np.ndarray:
    shape_values = list(shape or [])
    action_dim = action_dim_from_shape(shape_values)
    branch_mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if branch_mask.size != sum(shape_values):
        return np.ones((action_dim,), dtype=np.bool_)
    offsets = np.cumsum([0] + shape_values)
    joint = np.ones((action_dim,), dtype=np.bool_)
    for index in range(action_dim):
        action = unflatten_action(index, shape_values)
        for branch, value in enumerate(action):
            if not branch_mask[offsets[branch] + value]:
                joint[index] = False
                break
    if not np.any(joint):
        joint[0] = True
    return joint


def ensure_action_mask(
    mask: Optional[Sequence[Any]],
    action_dim: int,
    shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if mask is None:
        return np.ones((action_dim,), dtype=np.bool_)
    arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if shape is not None and len(list(shape)) > 1 and arr.size == sum(shape):
        return branch_mask_to_joint_mask(arr, shape)
    if arr.size != action_dim:
        arr = arr[:action_dim] if arr.size > action_dim else np.pad(arr, (0, action_dim - arr.size), constant_values=True)
    if not np.any(arr):
        arr[0] = True
    return arr.astype(np.bool_, copy=False)


def format_joint_action_distribution(counts: Sequence[float], shape: Sequence[int]) -> str:
    counts_arr = np.asarray(counts, dtype=np.float32).reshape(-1)
    total = float(counts_arr.sum())
    if total <= 0:
        return "no actions"
    shape_values = list(shape or [len(counts_arr)])
    branch_names = ("move", "look", "jump", "attack", "dash", "needle", "bind", "tool")
    value_names = (
        ("none", "left", "right"),
        ("none", "up", "down"),
        ("off", "on"),
        ("off", "on"),
        ("off", "on"),
        ("off", "on"),
        ("off", "on"),
        ("none", "neutral", "up", "down"),
    )
    branch_counts = [np.zeros(dim, dtype=np.float64) for dim in shape_values]
    top_indices = np.argsort(counts_arr)[-5:][::-1]
    for index, count in enumerate(counts_arr):
        if count <= 0:
            continue
        action = unflatten_action(index, shape_values)
        for branch, value in enumerate(action):
            branch_counts[branch][value] += float(count)
    pieces = []
    for branch, values in enumerate(branch_counts):
        labels = value_names[branch] if branch < len(value_names) else tuple(str(i) for i in range(len(values)))
        dist = "/".join(
            f"{labels[i] if i < len(labels) else i}:{values[i] / total * 100:.1f}%"
            for i in range(len(values))
        )
        pieces.append(f"{branch_names[branch] if branch < len(branch_names) else branch}({dist})")
    top = ", ".join(
        f"{unflatten_action(int(idx), shape_values)}:{counts_arr[idx] / total * 100:.1f}%"
        for idx in top_indices
        if counts_arr[idx] > 0
    )
    return "; ".join(pieces) + (f"; top {top}" if top else "")


def augment_state(state: Sequence[float]) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if future_predictor is None:
        return state_arr
    return future_predictor.augment_observation(state_arr)


def augment_observation_batch(observations: np.ndarray) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    if future_predictor is None:
        return obs
    return future_predictor.augment_observations(obs)


def extract_boss_hp_percent(obs: Sequence[float]) -> float:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size <= BOSS_HP_OBS_INDEX:
        return float("nan")
    return float(np.clip(arr[BOSS_HP_OBS_INDEX], 0.0, 1.0) * 100.0)


def extract_phase(obs: Sequence[float]) -> float:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < BOSS_PHASE_OBS_OFFSET + BOSS_PHASE_OBS_SIZE:
        return 1.0
    phase_one_hot = arr[BOSS_PHASE_OBS_OFFSET:BOSS_PHASE_OBS_OFFSET + BOSS_PHASE_OBS_SIZE]
    return float(int(np.argmax(phase_one_hot)) + 1)


def initialize_model(
    obs_size: int,
    boss_name: str,
    action_space_shape: List[int] = None,
    observation_type: str = "vector",
    vector_obs_size: int = None,
    visual_w: int = 0,
    visual_h: int = 0,
    is_eval: bool = False,
) -> Dict[str, Any]:
    global model, unity_obs_dim, obs_dim, action_shape, current_boss, eval_mode
    global future_predictor, reward_normalizer, agent_trace_recorder

    if observation_type != "vector":
        raise ValueError("DQN backend currently supports vector observations only")
    unity_obs_dim = int(obs_size)
    current_boss = boss_name
    action_shape = list(action_space_shape or [])
    if not action_shape:
        raise ValueError("DQN backend requires a non-empty action space shape")
    action_dim = action_dim_from_shape(action_shape)
    eval_mode = bool(is_eval)
    reward_normalizer = RunningRewardNormalizer()
    future_predictor = FuturePredictorRuntime.try_load(boss_name) if PREDICTOR_ENABLED else None
    if future_predictor is not None and future_predictor.input_dim != unity_obs_dim:
        print(
            f"[FuturePredictor] Skipping incompatible predictor input "
            f"{future_predictor.input_dim}; current obs is {unity_obs_dim}"
        )
        future_predictor = None
    predictor_feature_dim = future_predictor.feature_dim if future_predictor is not None else 0
    obs_dim = unity_obs_dim + predictor_feature_dim
    model_name = model_key_for_action_space(boss_name, action_shape, unity_obs_dim, predictor_feature_dim)
    model = DQNAgent(
        boss_name=boss_name,
        model_name=model_name,
        obs_size=obs_dim,
        unity_obs_size=unity_obs_dim,
        action_dim=action_dim,
        action_shape=action_shape,
        predictor_feature_dim=predictor_feature_dim,
        eval_mode=eval_mode,
    )

    os.makedirs(model.model_dir, exist_ok=True)
    candidates = [
        ("best", os.path.join(model.model_dir, "best_checkpoint.pt")),
        ("latest", os.path.join(model.model_dir, "checkpoint.pt")),
    ]
    checkpoint_loaded = False
    loaded_kind = None
    loaded_path = None
    for kind, path in candidates:
        if not os.path.exists(path):
            continue
        try:
            model.load(path)
            checkpoint_loaded = True
            loaded_kind = kind
            loaded_path = path
            print(f"[DQN] Loaded {kind} checkpoint: {path}")
            break
        except Exception as exc:
            print(f"[DQN] Failed to load {kind} checkpoint {path}: {exc}")

    if not checkpoint_loaded:
        model.save("checkpoint")

    if agent_trace_recorder is not None:
        agent_trace_recorder.close("reinitialize")
    agent_trace_recorder = None
    if AGENT_TRACE_ENABLED and not eval_mode:
        agent_trace_recorder = AgentTraceRecorder(AGENT_TRACE_ROOT, AGENT_TRACE_STEP_SECONDS, AGENT_TRACE_MIN_STEPS)
        agent_trace_recorder.start_boss(boss_name, action_shape)
        print(f"[DQN] Agent trace recording enabled: {AGENT_TRACE_ROOT}, step_seconds={AGENT_TRACE_STEP_SECONDS}")

    if DQN_PREFILL_ENABLED and not eval_mode:
        prefill_replay(model, boss_name, unity_obs_dim, action_shape, DQN_PREFILL_MAX_SAMPLES)

    print(
        f"[DQN] Initialized: obs={obs_dim} unity_obs={unity_obs_dim} action_shape={action_shape} joint_actions={action_dim} "
        f"buffer={DQN_BUFFER_SIZE} elite_buffer={DQN_ELITE_BUFFER_SIZE} "
        f"elite_batch={DQN_ELITE_BATCH_FRACTION:.2f} n_step={DQN_N_STEP} gamma={DQN_GAMMA} lr={DQN_LR}"
    )
    return {
        "initialized": True,
        "boss_name": boss_name,
        "observation_size": obs_dim,
        "checkpoint_loaded": checkpoint_loaded,
        "loaded_checkpoint_kind": loaded_kind,
        "loaded_checkpoint_path": loaded_path,
    }


def get_action(state: List[float], action_mask: Optional[List[int]] = None) -> List[int]:
    if model is None or unity_obs_dim is None:
        raise ValueError("DQN model not initialized")
    if len(state) != unity_obs_dim:
        raise ValueError(f"Expected Unity obs size {unity_obs_dim}, got {len(state)}")
    obs = augment_state(state)
    mask = ensure_action_mask(action_mask, model.action_dim, model.action_shape)
    action_id = model.select_action(obs, mask)
    return unflatten_action(action_id, model.action_shape)


def store_transition(
    state: List[float],
    action: List[int],
    reward: float,
    next_state: List[float],
    done: bool,
    action_mask: Optional[List[int]] = None,
    next_action_mask: Optional[List[int]] = None,
    reward_components: Optional[Dict[str, Any]] = None,
    teacher_action: Optional[List[int]] = None,
) -> None:
    if model is None or reward_normalizer is None or unity_obs_dim is None:
        raise ValueError("DQN model not initialized")
    if len(state) != unity_obs_dim or len(next_state) != unity_obs_dim:
        raise ValueError("Observation size mismatch")
    action_id = flatten_action(action, model.action_shape)
    obs = augment_state(state)
    next_obs = augment_state(next_state)
    mask = ensure_action_mask(action_mask, model.action_dim, model.action_shape)
    next_mask = ensure_action_mask(next_action_mask if next_action_mask is not None else action_mask, model.action_dim, model.action_shape)
    normalized_reward = reward_normalizer.normalize(float(reward)) if DQN_REWARD_NORM_ENABLED else float(reward)

    if agent_trace_recorder is not None:
        agent_trace_recorder.record(
            state=state,
            action=action,
            reward_raw=float(reward),
            reward_normalized=normalized_reward,
            done=done,
            action_mask=action_mask,
            reward_components=reward_components,
            teacher_action=teacher_action,
        )

    model.store_transition(obs, action_id, normalized_reward, next_obs, done, mask, next_mask)


def prefill_replay(agent: DQNAgent, boss_name: str, obs_size: int, prefill_action_shape: Sequence[int], max_samples: int) -> None:
    roots = prefill_roots(boss_name)
    added = 0
    elite_added = 0
    seen_files = 0
    print(
        f"[DQN] Prefill scanning roots={roots}, max_samples={max_samples}, "
        f"max_files={DQN_PREFILL_MAX_FILES}, include_agent_traces={DQN_PREFILL_INCLUDE_AGENT_TRACES}"
    )
    for path in iter_jsonl_files(roots):
        if added >= max_samples:
            break
        if DQN_PREFILL_MAX_FILES > 0 and seen_files >= DQN_PREFILL_MAX_FILES:
            break
        seen_files += 1
        if seen_files == 1 or seen_files % 25 == 0:
            print(f"[DQN] Prefill progress: files={seen_files}, transitions={added}, current={path}")
        records = _load_file_records(path)
        if len(records) < 2:
            continue
        file_boss = records[0].get("boss_name", "")
        if _normalize(file_boss) != _normalize(boss_name):
            continue
        adapted = _adapt_observation_sequence(records, obs_size)
        file_is_elite = is_elite_episode_observations(adapted)
        for index in range(len(records) - 1):
            if added >= max_samples:
                break
            obs = adapted[index]
            next_obs = adapted[index + 1]
            if not isinstance(obs, list) or not isinstance(next_obs, list):
                continue
            action = _adapt_action(records[index].get("action", []), obs, list(prefill_action_shape), records[index])
            if action is None:
                continue
            mask = _adapt_mask(records[index].get("action_mask"), obs, list(prefill_action_shape), records[index])
            next_mask = _adapt_mask(records[index + 1].get("action_mask"), next_obs, list(prefill_action_shape), records[index + 1])
            action_mask = ensure_action_mask(mask, agent.action_dim, agent.action_shape)
            next_action_mask = ensure_action_mask(next_mask, agent.action_dim, agent.action_shape)
            reward = prefill_reward(records[index], obs, next_obs)
            done = bool(records[index].get("done", False))
            replay_transition = ReplayTransition(
                obs=augment_state(obs),
                action=flatten_action(action, agent.action_shape),
                reward=reward,
                next_obs=augment_state(next_obs),
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
                discount=0.0 if done else DQN_GAMMA,
                priority=prefill_priority(records[index], obs, next_obs, reward),
            )
            add_replay_transition(agent.replay, replay_transition)
            if file_is_elite:
                add_replay_transition(agent.elite_replay, replay_transition)
                elite_added += 1
            added += 1
    if added:
        print(f"[DQN] Prefilled replay with {added} transitions from {seen_files} files, elite {elite_added}")
    else:
        print(f"[DQN] Replay prefill found no matching transitions in {roots} ({seen_files} files scanned)")


def prefill_roots(boss_name: str) -> List[str]:
    configured = os.environ.get("SILKSONGRL_DQN_PREFILL_DIRS")
    if configured:
        return [part for part in configured.split(os.pathsep) if part]
    boss_key = _normalize(boss_name)
    roots = [os.path.join(DEFAULT_DEMO_ROOT, boss_key.replace("lace_boss2_new", "lace_2"))]
    if DQN_PREFILL_INCLUDE_AGENT_TRACES:
        roots.append(os.path.join(AGENT_TRACE_ROOT, boss_key))
    return roots


def iter_jsonl_files(roots: Iterable[str]) -> Iterable[Path]:
    for root_value in roots:
        root = Path(root_value)
        if root.exists():
            yield from sorted(root.rglob("*.jsonl"))


def is_elite_episode_observations(observations: Sequence[Any]) -> bool:
    max_phase = 1.0
    min_boss_hp = float("inf")
    for obs in observations:
        if not isinstance(obs, list):
            continue
        max_phase = max(max_phase, extract_phase(obs))
        boss_hp = extract_boss_hp_percent(obs)
        if np.isfinite(boss_hp):
            min_boss_hp = min(min_boss_hp, boss_hp)
    return max_phase >= DQN_ELITE_MIN_PHASE or min_boss_hp < DQN_ELITE_MAX_BOSS_HP_PERCENT


def prefill_reward(record: dict, obs: Sequence[float], next_obs: Sequence[float]) -> float:
    for key in ("reward_normalized", "reward"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(np.clip(value, -DQN_REWARD_NORM_CLIP, DQN_REWARD_NORM_CLIP))
    boss_delta = (float(obs[BOSS_HP_OBS_INDEX]) - float(next_obs[BOSS_HP_OBS_INDEX])) * 100.0
    hero_delta = (float(obs[4]) - float(next_obs[4])) * 100.0
    phase_delta = max(0.0, extract_phase(next_obs) - extract_phase(obs))
    reward = boss_delta * 0.25 - hero_delta * 0.08 + phase_delta * 5.0
    if record.get("done") and float(next_obs[BOSS_HP_OBS_INDEX]) <= 0.01:
        reward += 10.0
    elif record.get("done"):
        reward -= 5.0
    return float(np.clip(reward, -DQN_REWARD_NORM_CLIP, DQN_REWARD_NORM_CLIP))


def prefill_priority(record: dict, obs: Sequence[float], next_obs: Sequence[float], reward: float) -> float:
    boss_hp = extract_boss_hp_percent(next_obs)
    phase = extract_phase(next_obs)
    priority = abs(float(reward)) + 1.0
    if phase >= 2:
        priority += 2.0
    if boss_hp <= 60.0:
        priority += 2.0
    if record.get("source", "").startswith("human"):
        priority += 1.0
    return priority
