import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bc_loss import parse_action_value_weights, parse_branch_weights, parse_nonzero_boosts
from frame_encoder import FrameStackFeatureExtractor
from sb3_ppo_override import CustomPPO, MODEL_ROOT
from demo_dataset import DEFAULT_DEMO_ROOT, load_demonstrations
from future_predictor import FuturePredictorRuntime

model: Optional[CustomPPO] = None
obs_dim: Optional[int] = None
unity_obs_dim: Optional[int] = None
action_shape: Optional[List[int]] = None
current_boss: Optional[str] = None
obs_type: Optional[str] = None          # 'vector' or 'hybrid'
vector_obs_dim: Optional[int] = None    # Size of vector portion
visual_width: Optional[int] = None      # Width of visual observation (0 if vector-only)
visual_height: Optional[int] = None     # Height of visual observation (0 if vector-only)
eval_mode: bool = False

BC_COEF = float(os.environ.get("SILKSONGRL_BC_COEF", "0.000"))
BC_DECAY = float(os.environ.get("SILKSONGRL_BC_DECAY", "0.99"))
BC_BATCH_SIZE = int(os.environ.get("SILKSONGRL_BC_BATCH_SIZE", "512"))
BC_MIN_STEPS = int(os.environ.get("SILKSONGRL_BC_MIN_STEPS", "20"))
BC_MAX_SAMPLES = os.environ.get("SILKSONGRL_BC_MAX_SAMPLES")
BC_MAX_SAMPLES = int(BC_MAX_SAMPLES) if BC_MAX_SAMPLES else None
BC_RESPECT_MASKS = os.environ.get("SILKSONGRL_BC_RESPECT_MASKS", "0") == "1"
BC_BRANCH_WEIGHTS = os.environ.get("SILKSONGRL_BC_BRANCH_WEIGHTS")
BC_NONZERO_BOOSTS = os.environ.get("SILKSONGRL_BC_NONZERO_BOOSTS", os.environ.get("SILKSONGRL_BC_NONZERO_BOOST"))
BC_ACTION_VALUE_WEIGHTS = os.environ.get("SILKSONGRL_BC_ACTION_VALUE_WEIGHTS")

PPO_N_STEPS = int(os.environ.get("SILKSONGRL_PPO_N_STEPS", "1024"))
PPO_BATCH_SIZE = int(os.environ.get("SILKSONGRL_PPO_BATCH_SIZE", "256"))
PPO_LEARNING_RATE = float(os.environ.get("SILKSONGRL_PPO_LEARNING_RATE", "1e-5"))
PPO_ENT_COEF = float(os.environ.get("SILKSONGRL_PPO_ENT_COEF", "0.0015"))
PPO_GAMMA = float(os.environ.get("SILKSONGRL_PPO_GAMMA", "0.9995"))
USE_FRAME_ENCODER = os.environ.get("SILKSONGRL_USE_FRAME_ENCODER", "1") != "0"
FRAME_STACK_SIZE = int(os.environ.get("SILKSONGRL_FRAME_STACK", "16"))
ADVISOR_COEF = float(os.environ.get("SILKSONGRL_ADVISOR_COEF", "0.0"))
ADVISOR_BATCH_SIZE = int(os.environ.get("SILKSONGRL_ADVISOR_BATCH_SIZE", "256"))
ADVISOR_BUFFER_SIZE = int(os.environ.get("SILKSONGRL_ADVISOR_BUFFER_SIZE", "50000"))
REWARD_NORM_ENABLED = os.environ.get("SILKSONGRL_REWARD_NORM", "1") != "0"
REWARD_NORM_CLIP = float(os.environ.get("SILKSONGRL_REWARD_NORM_CLIP", "10"))
PREDICTOR_ENABLED = os.environ.get("SILKSONGRL_FUTURE_PREDICTOR", "0") != "0"
AGENT_TRACE_ENABLED = os.environ.get("SILKSONGRL_SAVE_AGENT_TRACES", "0") != "0"
AGENT_TRACE_ROOT = os.environ.get(
    "SILKSONGRL_AGENT_TRACE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_traces"),
)
AGENT_TRACE_STEP_SECONDS = float(os.environ.get("SILKSONGRL_AGENT_TRACE_STEP_SECONDS", "0.05"))
AGENT_TRACE_MIN_STEPS = int(os.environ.get("SILKSONGRL_AGENT_TRACE_MIN_STEPS", "20"))

future_predictor: Optional[FuturePredictorRuntime] = None
agent_trace_recorder: Optional["AgentTraceRecorder"] = None


class AgentTraceRecorder:
    def __init__(
        self,
        root: str,
        step_seconds: float = 0.05,
        min_steps: int = 20,
    ) -> None:
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
        self.boss_key = self._safe_key(boss_name)
        self.action_space_shape = list(action_space_shape or [])

    def record(
        self,
        state: List[float],
        action: List[int],
        reward_raw: float,
        reward_normalized: float,
        done: bool,
        action_mask: Optional[List[int]] = None,
        reward_components: Optional[Any] = None,
        teacher_action: Optional[List[int]] = None,
    ) -> None:
        if not self.boss_name or not self.boss_key:
            return

        self._ensure_open()
        if self.file is None:
            return

        record = {
            "version": 8,
            "source": "ppo_agent",
            "action_space_shape": self.action_space_shape,
            "boss_key": self.boss_key,
            "boss_name": self.boss_name,
            "episode_id": self.episode_id,
            "step": self.step,
            "time": self.step * self.step_seconds,
            "done": bool(done),
            "terminal_reason": "done" if done else "",
            "observation_size": len(state),
            "observation": self._jsonable_list(state, float),
            "action": self._jsonable_list(action, int),
            "action_mask": self._jsonable_list(action_mask, int) if action_mask is not None else None,
            "teacher_action": self._jsonable_list(teacher_action, int) if teacher_action is not None else None,
            "reward_raw": float(reward_raw),
            "reward_normalized": float(reward_normalized),
            "reward_components": self._jsonable(reward_components),
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
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        millis = int((time.time() % 1.0) * 1000)
        self.episode_id = f"{timestamp}_{millis:03d}"
        self.path = os.path.join(folder, f"{self.episode_id}.jsonl")
        self.file = open(self.path, "w", encoding="utf-8")
        self.step = 0

    @staticmethod
    def _safe_key(value: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")

    @classmethod
    def _jsonable_list(cls, values: Any, cast: Any) -> List[Any]:
        if values is None:
            return []
        return [cast(value) for value in list(values)]

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): cls._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


class RunningRewardNormalizer:
    def __init__(self, epsilon: float = 1e-4) -> None:
        self.count = float(epsilon)
        self.mean = 0.0
        self.m2 = 1.0 * float(epsilon)

    @property
    def variance(self) -> float:
        return max(self.m2 / max(self.count, 1.0), 1e-6)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def normalize(self, value: float) -> float:
        value = float(value)
        delta = value - self.mean
        self.count += 1.0
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        normalized = (value - self.mean) / self.std
        return float(np.clip(normalized, -REWARD_NORM_CLIP, REWARD_NORM_CLIP))


reward_normalizer = RunningRewardNormalizer()


class DummyEnv(gym.Env):
    """
    Minimal env to satisfy SB3.
    Supports both vector-only and hybrid (vector + visual) observation modes.
    """

    def __init__(
        self, 
        obs_size: int, 
        action_space_shape: List[int] = None,
        observation_type: str = "vector",
        vector_obs_size: int = None,
        visual_w: int = 0,
        visual_h: int = 0
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.observation_type = observation_type
        self.vector_obs_size = vector_obs_size if vector_obs_size else obs_size
        self.visual_width = visual_w
        self.visual_height = visual_h
        
        if observation_type == "hybrid" and visual_w > 0 and visual_h > 0:
            # Dict observation space for hybrid: separate vector and visual
            self.observation_space = spaces.Dict({
                "vector": spaces.Box(0.0, 1.0, shape=(self.vector_obs_size,), dtype=np.float32),
                "visual": spaces.Box(0, 255, shape=(1, visual_h, visual_w), dtype=np.uint8),  # (C, H, W)
            })
        else:
            # Flat observation space for vector-only
            self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)
        
        self.action_space = spaces.MultiDiscrete(action_space_shape)

    def _make_obs(self) -> Any:
        """Create a zero observation matching the observation space."""
        if self.observation_type == "hybrid" and self.visual_width > 0:
            return {
                "vector": np.zeros(self.vector_obs_size, dtype=np.float32),
                "visual": np.zeros((1, self.visual_height, self.visual_width), dtype=np.uint8),
            }
        else:
            return np.zeros(self.obs_size, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        return self._make_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self._make_obs(), 0.0, True, False, {}


def normalize_boss_name(boss_name: str) -> str:
    return boss_name.replace(" ", "_").lower()


def model_key_for_action_space(
    boss_name: str,
    action_space_shape: Optional[List[int]],
    observation_size: Optional[int] = None,
    predictor_feature_dim: int = 0,
) -> str:
    key = normalize_boss_name(boss_name)
    if list(action_space_shape or []) in ([11], [26]):
        suffix = "_move_offense26" if list(action_space_shape or []) == [26] else "_option11"
        if observation_size in (2144, 2192):
            suffix += "_fsmsemantic80_facing"
        elif observation_size == 2112:
            suffix += "_fsmsemantic80"
        elif observation_size is not None and observation_size >= 4928:
            suffix += "_fsmelapsed"
        if observation_size == 2192:
            suffix += "_sparsephase"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        if USE_FRAME_ENCODER:
            suffix += "_frameenc"
        if observation_size is not None and observation_size >= 4000:
            suffix += f"_obs{observation_size}"
        return f"{key}{suffix}"
    if list(action_space_shape or []) == [13]:
        suffix = "_option13"
        if observation_size == 2144:
            suffix += "_fsmsemantic80_facing"
        elif observation_size == 2112:
            suffix += "_fsmsemantic80"
        elif observation_size is not None and observation_size >= 4928:
            suffix += "_fsmelapsed"
        if USE_FRAME_ENCODER:
            suffix += "_frameenc"
        return f"{key}{suffix}"
    if list(action_space_shape or []) in ([16], [30]):
        suffix = "_lowlevel30_raycast32_hitbox12" if list(action_space_shape or []) == [30] else "_macro16"
        if observation_size is not None and observation_size >= 4928:
            suffix += "_fsmelapsed"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        if USE_FRAME_ENCODER:
            suffix += "_frameenc"
        if observation_size is not None and observation_size >= 4000:
            suffix += f"_obs{observation_size}"
        return f"{key}{suffix}"
    if list(action_space_shape or []) == [3, 3, 2, 2, 2, 2, 2, 4]:
        suffix = "_rawkeys8"
        if observation_size is not None and observation_size >= 4928:
            suffix += "_fsmelapsed"
        if predictor_feature_dim > 0:
            suffix += f"_pred{predictor_feature_dim}"
        if USE_FRAME_ENCODER:
            suffix += "_frameenc"
        if observation_size is not None and observation_size >= 4000:
            suffix += f"_obs{observation_size}"
        return f"{key}{suffix}"
    return key


def initialize_model(
    obs_size: int, 
    boss_name: str, 
    action_space_shape: List[int] = None,
    observation_type: str = "vector",
    vector_obs_size: int = None,
    visual_w: int = 0,
    visual_h: int = 0,
    is_eval: bool = False
) -> Dict[str, Any]:
    """Initialize or load model; returns metadata about the initialization."""
    global model, obs_dim, unity_obs_dim, action_shape, current_boss, obs_type, vector_obs_dim
    global visual_width, visual_height
    global eval_mode, future_predictor, reward_normalizer, agent_trace_recorder

    unity_obs_dim = obs_size
    current_boss = boss_name
    obs_type = observation_type
    vector_obs_dim = vector_obs_size
    visual_width = visual_w
    visual_height = visual_h
    action_shape = action_space_shape
    eval_mode = is_eval
    model = None
    reward_normalizer = RunningRewardNormalizer()
    if agent_trace_recorder is not None:
        agent_trace_recorder.close("reinitialize")
        agent_trace_recorder = None
    future_predictor = FuturePredictorRuntime.try_load(boss_name) if PREDICTOR_ENABLED and observation_type == "vector" else None
    if future_predictor is not None and future_predictor.input_dim != obs_size:
        print(
            f"[FuturePredictor] Skipping incompatible predictor input "
            f"{future_predictor.input_dim}; current obs is {obs_size}"
        )
        future_predictor = None
    predictor_feature_dim = future_predictor.feature_dim if future_predictor is not None else 0
    obs_dim = obs_size + predictor_feature_dim

    normalized_boss_name = model_key_for_action_space(boss_name, action_space_shape, obs_size, predictor_feature_dim)
    model_dir = os.path.join(MODEL_ROOT, normalized_boss_name)
    checkpoint_path = os.path.join(model_dir, "checkpoint.zip")
    best_checkpoint_path = os.path.join(model_dir, "best_checkpoint.zip")

    print(f"[RLCore] Observation type: {obs_type}")
    print(f"[RLCore] Unity obs size: {unity_obs_dim}, Model obs size: {obs_dim}, Vector obs size: {vector_obs_dim}")
    if future_predictor is not None:
        print(f"[RLCore] Future predictor enabled: features={predictor_feature_dim}")
    if obs_type == "hybrid":
        print(f"[RLCore] Visual obs: {visual_width}x{visual_height} ({visual_width * visual_height})")
    print(f"[RLCore] Action space shape: {action_space_shape}")
    print(f"[RLCore] Eval mode: {eval_mode}")
    if AGENT_TRACE_ENABLED and not is_eval:
        agent_trace_recorder = AgentTraceRecorder(
            AGENT_TRACE_ROOT,
            step_seconds=AGENT_TRACE_STEP_SECONDS,
            min_steps=AGENT_TRACE_MIN_STEPS,
        )
        agent_trace_recorder.start_boss(boss_name, action_space_shape)
        print(
            f"[RLCore] Agent trace recording enabled: root={AGENT_TRACE_ROOT}, "
            f"step_seconds={AGENT_TRACE_STEP_SECONDS}"
        )

    env = DummyEnv(
        obs_dim,
        action_space_shape,
        observation_type=observation_type,
        vector_obs_size=obs_dim if observation_type == "vector" else vector_obs_size,
        visual_w=visual_w,
        visual_h=visual_h
    )
    

    if observation_type == "hybrid" and visual_w > 0:
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            net_arch=[256, 256, 128],
        )
    else:
        policy = "MlpPolicy"
        if USE_FRAME_ENCODER and observation_type == "vector" and obs_size % FRAME_STACK_SIZE == 0:
            policy_kwargs = dict(
                features_extractor_class=FrameStackFeatureExtractor,
                features_extractor_kwargs=dict(
                    frame_stack=FRAME_STACK_SIZE,
                    per_frame_features=64,
                    features_dim=512,
                    stacked_obs_dim=obs_size,
                    extra_features_dim=predictor_feature_dim,
                ),
                net_arch=[512, 256],
            )
            print(
                f"[RLCore] Using frame encoder: frame_stack={FRAME_STACK_SIZE}, "
                f"single_frame={obs_size // FRAME_STACK_SIZE}, features_dim=512"
            )
        else:
            policy_kwargs = dict(net_arch=[256, 256, 128])

    checkpoint_loaded = False
    loaded_checkpoint_kind = None
    loaded_checkpoint_path = None
    checkpoint_candidates = []
    if os.path.exists(best_checkpoint_path):
        checkpoint_candidates.append(("best", best_checkpoint_path))
    if os.path.exists(checkpoint_path):
        checkpoint_candidates.append(("latest", checkpoint_path))

    for checkpoint_kind, candidate_path in checkpoint_candidates:
        print(f"[RLCore] Loading {checkpoint_kind} checkpoint: {candidate_path}")
        try:
            model = CustomPPO.load(
                candidate_path,
                env=env,
                device="cpu",
            )
            checkpoint_loaded = True
            loaded_checkpoint_kind = checkpoint_kind
            loaded_checkpoint_path = candidate_path
            break
        except Exception as exc:
            print(f"[RLCore] {checkpoint_kind} checkpoint incompatible with current MaskablePPO setup: {exc}")
            model = None

    if checkpoint_candidates and model is None:
        print("[RLCore] No compatible checkpoint candidate found; initializing a fresh model instead")

    if model is None:
        print(f"[RLCore] No compatible checkpoint found, initializing fresh model")
        print(f"[RLCore] Using policy: {policy}")
        model = CustomPPO(
            policy,
            env,
            boss_name=normalized_boss_name,
            device="cpu",
            verbose=1,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            learning_rate=PPO_LEARNING_RATE,
            ent_coef=PPO_ENT_COEF,
            clip_range=0.2,
            n_epochs=10,
            gamma=PPO_GAMMA,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
        )
        checkpoint_loaded = False
        
        os.makedirs(model_dir, exist_ok=True)
        model.save(checkpoint_path.replace(".zip", ""))
        print(f"[RLCore] Saved initial checkpoint: {checkpoint_path}")

    if model is not None:
        model.set_advisor_config(
            coef=0.0 if is_eval else ADVISOR_COEF,
            batch_size=ADVISOR_BATCH_SIZE,
            buffer_size=ADVISOR_BUFFER_SIZE,
            respect_masks=True,
        )

    _configure_bc_auxiliary(boss_name, obs_size, action_space_shape, is_eval)

    return {
        "initialized": True,
        "boss_name": boss_name,
        "observation_size": obs_dim,
        "checkpoint_loaded": checkpoint_loaded,
        "loaded_checkpoint_kind": loaded_checkpoint_kind,
        "loaded_checkpoint_path": loaded_checkpoint_path,
    }


def _configure_bc_auxiliary(
    boss_name: str,
    obs_size: int,
    action_space_shape: List[int],
    is_eval: bool,
) -> None:
    if model is None or is_eval or BC_COEF <= 0:
        return

    try:
        dataset = load_demonstrations(
            demo_dir=DEFAULT_DEMO_ROOT,
            boss_name=boss_name,
            obs_size=unity_obs_dim or obs_size,
            action_shape=action_space_shape,
            min_steps=BC_MIN_STEPS,
            max_samples=BC_MAX_SAMPLES,
            drop_mask_violations=BC_RESPECT_MASKS,
        )
    except Exception as exc:
        print(f"[RLCore] BC auxiliary loss disabled: {exc}")
        return

    model.set_bc_dataset(
        augment_observation_batch(dataset.observations),
        dataset.actions,
        dataset.action_masks,
        coef=BC_COEF,
        decay=BC_DECAY,
        batch_size=BC_BATCH_SIZE,
        respect_masks=BC_RESPECT_MASKS,
        branch_weights=parse_branch_weights(BC_BRANCH_WEIGHTS, len(action_space_shape)),
        nonzero_boosts=parse_nonzero_boosts(BC_NONZERO_BOOSTS, len(action_space_shape)),
        action_value_weights=parse_action_value_weights(BC_ACTION_VALUE_WEIGHTS, action_space_shape),
    )
    print(f"[RLCore] Loaded BC demos from {len(dataset.files)} files for '{boss_name}'")


def _convert_to_obs(state: List[float]) -> Any:
    """Convert flat state array to observation format (flat or dict)."""
    if obs_type == "hybrid" and visual_width > 0:
        state_arr = np.array(state, dtype=np.float32)
        return {
            "vector": state_arr[:vector_obs_dim],
            "visual": (
                (np.clip(state_arr[vector_obs_dim:], 0.0, 1.0) * 255.0)
                .astype(np.uint8)
                .reshape(1, visual_height, visual_width)
            ),  # (C, H, W)
        }
    else:
        return np.array(state, dtype=np.float32)


def augment_state(state: List[float]) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32)
    if future_predictor is None:
        return state_arr

    return future_predictor.augment_observation(state_arr)


def augment_observation_batch(observations: np.ndarray) -> np.ndarray:
    if future_predictor is None:
        return observations

    return future_predictor.augment_observations(np.asarray(observations, dtype=np.float32))


def _convert_action_mask(action_mask: Optional[List[int]]) -> Optional[np.ndarray]:
    if action_mask is None:
        return None
    if action_shape is None:
        raise ValueError("Action space not initialized")

    expected_size = int(sum(action_shape))
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    if mask.size != expected_size:
        raise ValueError(f"Action mask size mismatch: expected {expected_size}, got {mask.size}")

    offset = 0
    for dim in action_shape:
        segment = mask[offset:offset + dim]
        if not np.any(segment):
            segment[0] = True
        offset += dim

    return mask


def get_action(state: List[float], action_mask: Optional[List[int]] = None) -> List[int]:
    if model is None:
        raise ValueError("Model not initialized")
    if len(state) != unity_obs_dim:
        raise ValueError(f"Expected Unity obs size {unity_obs_dim}, got {len(state)}")

    obs = _convert_to_obs(augment_state(state))
    mask = _convert_action_mask(action_mask)
    action, _ = model.predict(obs, deterministic=eval_mode, action_masks=mask)
    return action.tolist()


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
    if model is None:
        raise ValueError("Model not initialized")
    if len(state) != unity_obs_dim or len(next_state) != unity_obs_dim:
        raise ValueError("Observation size mismatch")
    if len(action) != len(action_shape):
        raise ValueError(f"Action size mismatch: expected {len(action_shape)}, got {len(action)}")
    if teacher_action is not None and len(teacher_action) != len(action_shape):
        raise ValueError(f"Teacher action size mismatch: expected {len(action_shape)}, got {len(teacher_action)}")

    obs = _convert_to_obs(augment_state(state))
    next_obs = _convert_to_obs(augment_state(next_state))
    mask = _convert_action_mask(action_mask)
    normalized_reward = reward_normalizer.normalize(float(reward)) if REWARD_NORM_ENABLED else float(reward)
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
    model.store_transition(
        obs,
        action,
        normalized_reward,
        next_obs,
        done,
        action_mask=mask,
        reward_components=reward_components,
        teacher_action=teacher_action,
    )

