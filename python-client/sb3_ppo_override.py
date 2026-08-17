import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from gymnasium import spaces
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import explained_variance
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from bc_loss import DEFAULT_BRANCH_WEIGHTS, DEFAULT_NONZERO_BOOSTS, weighted_multidiscrete_bc_loss
from demo_dataset import BASE_OBS_SIZE, LACE_COMPRESSED_FSM_SIZE


MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WIN_RATE_WINDOW = 25
CHECKPOINT_SAVE_FREQ = 25
BOSS_HP_OBS_INDEX = 9
ROLLBACK_BAD_WINDOW_COUNT = 2
ROLLBACK_BOSS_HP_TOLERANCE = 2.0
BEST_CHECKPOINT_PHASE_WEIGHT = 2.0
BEST_CHECKPOINT_SCORE_METRIC = "avg_reward_minus_boss_hp_plus_2x_avg_phase"
BOSS_PHASE_OBS_OFFSET = BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE
BOSS_PHASE_OBS_SIZE = 3
REWARD_COMPONENT_KEYS = [
    "damageReward",
    "healReward",
    "attackReward",
    "baseSurvivalReward",
    "dodgeReward",
    "resourceReward",
    "positionReward",
    "phaseReward",
    "heroDamagePenalty",
]
COMPOSITE26_ACTION_NAMES = [
    "KeepSweetSpot",
    "RetreatFromBossDodge",
    "DodgeOverBossHighJump",
    "CrossUnderAirBoss",
    "MoveToCenterSafely",
    "HoldSafeSide",
    "KeepSweetSpot+Slash",
    "Retreat+Slash",
    "DodgeOverBoss+Slash",
    "CrossUnder+Slash",
    "MoveToCenter+Slash",
    "Hold+Slash",
    "KeepSweetSpot+AntiAir",
    "Retreat+AntiAir",
    "DodgeOverBoss+AntiAir",
    "CrossUnder+AntiAir",
    "MoveToCenter+AntiAir",
    "Hold+AntiAir",
    "KeepSweetSpot+Tool",
    "Retreat+Tool",
    "DodgeOverBoss+Tool",
    "CrossUnder+Tool",
    "MoveToCenter+Tool",
    "Hold+Tool",
    "UseCloseSkill",
    "BindHeal",
]
LOW_LEVEL_ACTION_NAMES = [
    "Idle",
    "MoveLeft",
    "MoveRight",
    "LookUp",
    "LookDown",
    "Jump",
    "JumpLeft",
    "JumpRight",
    "Dash",
    "DashLeft",
    "DashRight",
    "Attack",
    "MoveLeftAttack",
    "MoveRightAttack",
    "UpAttack",
    "DownAttack",
    "JumpAttack",
    "JumpLeftAttack",
    "JumpRightAttack",
    "DashLeftAttack",
    "DashRightAttack",
    "Needle",
    "BindHeal",
    "ToolNeutral",
    "ToolUp",
    "ToolDown",
    "MoveLeftTool",
    "MoveRightTool",
    "JumpLeftTool",
    "JumpRightTool",
]
MACRO_ACTION_NAMES = COMPOSITE26_ACTION_NAMES
OPTION11_TO_COMPOSITE26 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 11,
    7: 17,
    8: 24,
    9: 23,
    10: 25,
}
LEGACY_MACRO16_TO_OPTION11 = {
    0: 5,   # Idle -> HoldSafeSide
    1: 0,   # ApproachBoss -> KeepSweetSpot
    2: 1,   # RetreatFromBoss -> RetreatFromBossDodge
    3: 0,   # KeepAttackRange -> KeepSweetSpot
    4: 4,   # MoveToCenter -> MoveToCenterSafely
    5: 2,   # Jump -> DodgeOverBossHighJump
    6: 1,   # DashAway -> RetreatFromBossDodge
    7: 2,   # DashThrough -> DodgeOverBossHighJump
    8: 6,   # HorizontalSlash -> PunishWithSlash
    9: 7,   # UpSlash -> AntiAirSlash
    10: 6,  # DownSlash -> PunishWithSlash
    11: 10, # BindHeal -> BindHeal
    12: 9,  # Needle -> UseRangedTool
    13: 8,  # ToolNeutral -> UseCloseSkill
    14: 9,  # ToolUp -> UseRangedTool
    15: 9,  # ToolDown -> UseRangedTool
}
LEGACY_OPTION13_TO_OPTION11 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 0,   # ApproachForBigPunish -> KeepSweetSpot
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 1,  # FindBindWindow -> RetreatFromBossDodge
    12: 10,
}


def zero_reward_components() -> Dict[str, float]:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def normalize_reward_components(components: Optional[Any]) -> Dict[str, float]:
    normalized = zero_reward_components()
    if not components:
        return normalized

    if isinstance(components, (list, tuple, np.ndarray)):
        if len(components) == 5 and len(REWARD_COMPONENT_KEYS) >= 8:
            legacy_keys = [
                "damageReward",
                "healReward",
                "attackReward",
                "baseSurvivalReward",
                "heroDamagePenalty",
            ]
            for index, key in enumerate(legacy_keys):
                try:
                    normalized[key] = float(components[index])
                except (IndexError, TypeError, ValueError):
                    normalized[key] = 0.0
            return normalized

        if len(components) == 8 and len(REWARD_COMPONENT_KEYS) == 9:
            legacy_keys = [
                "damageReward",
                "healReward",
                "attackReward",
                "baseSurvivalReward",
                "dodgeReward",
                "resourceReward",
                "positionReward",
                "heroDamagePenalty",
            ]
            for index, key in enumerate(legacy_keys):
                try:
                    normalized[key] = float(components[index])
                except (IndexError, TypeError, ValueError):
                    normalized[key] = 0.0
            return normalized

        for index, key in enumerate(REWARD_COMPONENT_KEYS):
            if index >= len(components):
                break
            try:
                normalized[key] = float(components[index])
            except (TypeError, ValueError):
                normalized[key] = 0.0
        return normalized

    for key in REWARD_COMPONENT_KEYS:
        try:
            normalized[key] = float(components.get(key, 0.0))
        except (TypeError, ValueError):
            normalized[key] = 0.0

    if "baseSurvivalReward" in normalized and normalized["baseSurvivalReward"] == 0.0:
        try:
            normalized["baseSurvivalReward"] = float(components.get("survivalReward", 0.0))
        except (TypeError, ValueError):
            normalized["baseSurvivalReward"] = 0.0

    return normalized


def add_reward_components(target: Dict[str, float], components: Dict[str, float]) -> None:
    for key in REWARD_COMPONENT_KEYS:
        target[key] = target.get(key, 0.0) + components.get(key, 0.0)


def average_reward_components(component_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not component_list:
        return zero_reward_components()

    return {
        key: float(np.mean([components.get(key, 0.0) for components in component_list]))
        for key in REWARD_COMPONENT_KEYS
    }


def format_reward_components(components: Dict[str, float]) -> str:
    return ", ".join(f"{key} {components.get(key, 0.0):.2f}" for key in REWARD_COMPONENT_KEYS)


def zero_action_counts() -> List[int]:
    return [0 for _ in MACRO_ACTION_NAMES]


def normalize_action_counts(counts: Optional[Any]) -> List[int]:
    normalized = zero_action_counts()
    if counts is None:
        return normalized

    try:
        values = list(counts)
    except TypeError:
        return normalized

    if not values:
        return normalized

    if len(values) == 11 and len(normalized) == 26:
        for legacy_index, composite_index in OPTION11_TO_COMPOSITE26.items():
            try:
                normalized[composite_index] += int(values[legacy_index])
            except (IndexError, TypeError, ValueError):
                continue
        return normalized

    if len(values) == 13 and len(normalized) == 26:
        for legacy_index, option_index in LEGACY_OPTION13_TO_OPTION11.items():
            try:
                normalized[OPTION11_TO_COMPOSITE26[option_index]] += int(values[legacy_index])
            except (IndexError, TypeError, ValueError):
                continue
        return normalized

    if len(values) == 16 and len(normalized) == 26:
        for legacy_index, option_index in LEGACY_MACRO16_TO_OPTION11.items():
            try:
                normalized[OPTION11_TO_COMPOSITE26[option_index]] += int(values[legacy_index])
            except (IndexError, TypeError, ValueError):
                continue
        return normalized

    for index, value in enumerate(values[:len(normalized)]):
        try:
            normalized[index] = int(value)
        except (TypeError, ValueError):
            normalized[index] = 0
    return normalized


def average_action_counts(count_list: List[List[int]]) -> List[float]:
    if not count_list:
        return [0.0 for _ in MACRO_ACTION_NAMES]

    normalized = np.asarray([normalize_action_counts(counts) for counts in count_list], dtype=np.float32)
    return normalized.mean(axis=0).tolist()


def format_action_distribution(counts: List[float]) -> str:
    total = float(sum(counts))
    if total <= 0.0:
        return "none"

    return ", ".join(
        f"{name} {count / total * 100.0:.1f}%"
        for name, count in zip(MACRO_ACTION_NAMES, counts)
    )


class CustomPPO(MaskablePPO):
    # Need to define times_trained, episodes_completed, episode_rewards, and save_freq here
    # even though they could just be kept as defaut at 0 because otherwise we cannot load
    # them into the model after loading from a checkpoint
    def __init__(
        self,
        *args: Any,
        boss_name: Optional[str] = None,
        save_freq: int = CHECKPOINT_SAVE_FREQ,
        times_trained: int = 0,
        episodes_completed: int = 0,
        episode_rewards: Optional[List[float]] = None,
        episode_wins: Optional[List[bool]] = None,
        episode_boss_hp_percent: Optional[List[float]] = None,
        episode_phases: Optional[List[float]] = None,
        episode_reward_components: Optional[List[Dict[str, float]]] = None,
        episode_action_counts: Optional[List[List[int]]] = None,
        best_avg_boss_hp_percent: Optional[float] = None,
        best_avg_boss_hp_episode: int = 0,
        best_checkpoint_score: Optional[float] = None,
        best_checkpoint_avg_reward: Optional[float] = None,
        best_checkpoint_avg_phase: Optional[float] = None,
        window_avg_boss_hp_history: Optional[List[Dict[str, float]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.last_done = False
        self.times_trained = times_trained
        self.boss_name = boss_name
        self.save_freq = CHECKPOINT_SAVE_FREQ
        self.episodes_completed = episodes_completed

        self.episode_rewards: List[float] = episode_rewards or []
        self.episode_wins: List[bool] = episode_wins or []
        self.episode_boss_hp_percent: List[float] = episode_boss_hp_percent or []
        self.episode_phases: List[float] = [
            float(phase) for phase in (episode_phases or [])
        ]
        self.episode_reward_components: List[Dict[str, float]] = [
            normalize_reward_components(components)
            for components in (episode_reward_components or [])
        ]
        self.episode_action_counts: List[List[int]] = [
            normalize_action_counts(counts)
            for counts in (episode_action_counts or [])
        ]
        self.best_avg_boss_hp_percent = float(best_avg_boss_hp_percent) if best_avg_boss_hp_percent is not None else float("inf")
        self.best_avg_boss_hp_episode = int(best_avg_boss_hp_episode)
        self.best_checkpoint_score = float(best_checkpoint_score) if best_checkpoint_score is not None else float("-inf")
        self.best_checkpoint_avg_reward = float(best_checkpoint_avg_reward) if best_checkpoint_avg_reward is not None else float("-inf")
        self.best_checkpoint_avg_phase = float(best_checkpoint_avg_phase) if best_checkpoint_avg_phase is not None else float("nan")
        self.best_checkpoint_score_metric = BEST_CHECKPOINT_SCORE_METRIC
        self.window_avg_boss_hp_history: List[Dict[str, float]] = window_avg_boss_hp_history or []
        self.current_episode_reward = 0.0
        self.current_episode_components = zero_reward_components()
        self.current_episode_action_counts = zero_action_counts()
        self.warned_missing_reward_components = False
        self.bc_observations = None
        self.bc_actions = None
        self.bc_action_masks = None
        self.bc_coef = 0.0
        self.bc_coef_decay = 1.0
        self.bc_batch_size = 256
        self.bc_respect_masks = False
        self.bc_branch_weights = DEFAULT_BRANCH_WEIGHTS
        self.bc_nonzero_boosts = DEFAULT_NONZERO_BOOSTS
        self.bc_action_value_weights = None
        self.advisor_observations = []
        self.advisor_actions = []
        self.advisor_action_masks = []
        self.advisor_coef = 0.0
        self.advisor_batch_size = 256
        self.advisor_buffer_size = 50000
        self.advisor_respect_masks = True

        # Initilalize logger or SB3 complains
        if not hasattr(self, '_logger') or self._logger is None:
            from stable_baselines3.common.logger import configure
            self._logger = configure()

        # Only reset buffer if it exists (it won't exist during .load())
        if hasattr(self, 'rollout_buffer') and self.rollout_buffer is not None:
            self.rollout_buffer.reset()
    

    @property
    def logger(self):
        return self._logger

    def _excluded_save_params(self) -> List[str]:
        excluded = super()._excluded_save_params()
        return excluded + [
            "bc_observations",
            "bc_actions",
            "bc_action_masks",
            "advisor_observations",
            "advisor_actions",
            "advisor_action_masks",
        ]

    # Override load method to sneak in our own custom variables
    # Frankly there may be a better way to do this but I'm tired and 
    # if I keep trying I might claw my eyes out
    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device = "auto",
        boss_name: Optional[str] = None,
        **kwargs: Any
    ) -> "CustomPPO":
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        boss_name = data.get("boss_name", None)
        save_freq = data.get("save_freq", CHECKPOINT_SAVE_FREQ)
        times_trained = data.get("times_trained", 0)
        episodes_completed = data.get("episodes_completed", 0)
        episode_rewards = data.get("episode_rewards", [])
        episode_wins = data.get("episode_wins", [])
        episode_boss_hp_percent = data.get("episode_boss_hp_percent", [])
        episode_phases = data.get("episode_phases", [])
        episode_reward_components = data.get("episode_reward_components", [])
        episode_action_counts = data.get("episode_action_counts", [])
        best_avg_boss_hp_percent = data.get("best_avg_boss_hp_percent", None)
        best_avg_boss_hp_episode = data.get("best_avg_boss_hp_episode", 0)
        best_checkpoint_score = data.get("best_checkpoint_score", None)
        best_checkpoint_avg_reward = data.get("best_checkpoint_avg_reward", None)
        best_checkpoint_avg_phase = data.get("best_checkpoint_avg_phase", None)
        window_avg_boss_hp_history = data.get("window_avg_boss_hp_history", [])

        policy_class = data["policy_class"]
        if not issubclass(policy_class, MaskableActorCriticPolicy):
            raise ValueError("Checkpoint was created with a non-maskable policy")

        model = cls(
            policy=policy_class,
            env=None,
            device=device,
            boss_name=boss_name,
            save_freq=save_freq,
            times_trained=times_trained,
            episodes_completed=episodes_completed,
            episode_rewards=episode_rewards,
            episode_wins=episode_wins,
            episode_boss_hp_percent=episode_boss_hp_percent,
            episode_phases=episode_phases,
            episode_reward_components=episode_reward_components,
            episode_action_counts=episode_action_counts,
            best_avg_boss_hp_percent=best_avg_boss_hp_percent,
            best_avg_boss_hp_episode=best_avg_boss_hp_episode,
            best_checkpoint_score=best_checkpoint_score,
            best_checkpoint_avg_reward=best_checkpoint_avg_reward,
            best_checkpoint_avg_phase=best_checkpoint_avg_phase,
            window_avg_boss_hp_history=window_avg_boss_hp_history,
            _init_setup_model=False
        )

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.save_freq = CHECKPOINT_SAVE_FREQ
        model.episode_reward_components = [
            normalize_reward_components(components)
            for components in getattr(model, "episode_reward_components", [])
        ]
        model.episode_action_counts = [
            normalize_action_counts(counts)
            for counts in getattr(model, "episode_action_counts", [])
        ]
        model.episode_phases = [
            float(phase) for phase in getattr(model, "episode_phases", [])
        ]
        model.current_episode_components = normalize_reward_components(
            getattr(model, "current_episode_components", None)
        )
        model.current_episode_action_counts = normalize_action_counts(
            getattr(model, "current_episode_action_counts", None)
        )
        if not hasattr(model, "window_avg_boss_hp_history") or model.window_avg_boss_hp_history is None:
            model.window_avg_boss_hp_history = []
        if not hasattr(model, "best_avg_boss_hp_percent"):
            model.best_avg_boss_hp_percent = float("inf")
        if not hasattr(model, "best_avg_boss_hp_episode"):
            model.best_avg_boss_hp_episode = 0
        old_metric = getattr(model, "best_checkpoint_score_metric", None)
        old_score = getattr(model, "best_checkpoint_score", float("-inf"))
        if not hasattr(model, "best_checkpoint_avg_reward"):
            model.best_checkpoint_avg_reward = float("-inf")
        if not hasattr(model, "best_checkpoint_avg_phase"):
            model.best_checkpoint_avg_phase = float("nan")
        if old_metric != BEST_CHECKPOINT_SCORE_METRIC:
            model.best_checkpoint_score_metric = BEST_CHECKPOINT_SCORE_METRIC
            if np.isfinite(model.best_avg_boss_hp_percent) and np.isfinite(model.best_checkpoint_avg_reward):
                avg_phase = (
                    float(model.best_checkpoint_avg_phase)
                    if np.isfinite(model.best_checkpoint_avg_phase)
                    else 1.0
                )
                model.best_checkpoint_avg_phase = avg_phase
                model.best_checkpoint_score = model._best_checkpoint_score(
                    model.best_avg_boss_hp_percent,
                    model.best_checkpoint_avg_reward,
                    avg_phase,
                )
        if not hasattr(model, "best_checkpoint_score") or not np.isfinite(model.best_checkpoint_score):
            model.best_checkpoint_score = float("-inf")

        model._setup_model()
        model.set_parameters(params, exact_match=False)

        return model


    def start_new_rollout(self) -> None:
        self.rollout_buffer.reset()


    def _boss_directory(self) -> str:
        boss_dir = self.boss_name
        return os.path.join(MODEL_ROOT, boss_dir)

    def set_bc_dataset(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        coef: float = 0.1,
        decay: float = 0.98,
        batch_size: int = 256,
        respect_masks: bool = False,
        branch_weights: Optional[List[float]] = None,
        nonzero_boosts: Optional[List[float]] = None,
        action_value_weights: Optional[List[Optional[List[float]]]] = None,
    ) -> None:
        self.bc_observations = np.asarray(observations, dtype=np.float32)
        self.bc_actions = np.asarray(actions, dtype=np.int64)
        self.bc_action_masks = np.asarray(action_masks, dtype=np.bool_) if action_masks is not None else None
        self.bc_coef = float(coef)
        self.bc_coef_decay = float(decay)
        self.bc_batch_size = int(batch_size)
        self.bc_respect_masks = bool(respect_masks)
        self.bc_branch_weights = list(branch_weights) if branch_weights is not None else DEFAULT_BRANCH_WEIGHTS
        self.bc_nonzero_boosts = list(nonzero_boosts) if nonzero_boosts is not None else DEFAULT_NONZERO_BOOSTS
        self.bc_action_value_weights = action_value_weights

        if len(self.bc_observations) != len(self.bc_actions):
            raise ValueError("BC observation/action count mismatch")

        print(
            f"[CustomPPO] BC auxiliary loss enabled: samples={len(self.bc_observations)}, "
            f"coef={self.bc_coef}, decay={self.bc_coef_decay}, batch_size={self.bc_batch_size}, "
            f"respect_masks={self.bc_respect_masks}, branch_weights={self.bc_branch_weights}, "
            f"nonzero_boosts={self.bc_nonzero_boosts}, action_value_weights={self.bc_action_value_weights}"
        )

    def set_advisor_config(
        self,
        coef: float = 0.05,
        batch_size: int = 256,
        buffer_size: int = 50000,
        respect_masks: bool = True,
    ) -> None:
        self.advisor_coef = float(coef)
        self.advisor_batch_size = int(batch_size)
        self.advisor_buffer_size = int(buffer_size)
        self.advisor_respect_masks = bool(respect_masks)
        if not hasattr(self, "advisor_observations") or self.advisor_observations is None:
            self.advisor_observations = []
        if not hasattr(self, "advisor_actions") or self.advisor_actions is None:
            self.advisor_actions = []
        if not hasattr(self, "advisor_action_masks") or self.advisor_action_masks is None:
            self.advisor_action_masks = []
        print(
            f"[CustomPPO] Online advisor loss: coef={self.advisor_coef}, "
            f"batch_size={self.advisor_batch_size}, buffer_size={self.advisor_buffer_size}, "
            f"respect_masks={self.advisor_respect_masks}"
        )
   

    def plot_rewards(self, save_dir: str) -> None:
        """Generate and save a plot of episode rewards."""

        if len(self.episode_rewards) == 0:
            raise ValueError(f"No rewards to plot for {self.boss_name}")
        
        plt.figure(figsize=(12, 6))

        rewards = np.asarray(self.episode_rewards, dtype=np.float32)
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        plt.plot(episodes, rewards, alpha=0.3, label='Episode Rewards', color='blue')
        
        window = self.save_freq // 2
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ma_start = window
        ma_episodes = list(range(ma_start, ma_start + len(moving_avg)))
        plt.plot(ma_episodes, moving_avg, label=f'{window}-Episode Moving Average Rewards', 
                color='red', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Training Progress - {self.boss_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(save_dir, 'training_rewards.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        """Convert observation (flat array or dict) to tensor for policy."""
        if isinstance(obs, dict):
            tensors = {}
            for key, value in obs.items():
                tensor = torch.as_tensor(value).float().to(self.device)
                space = self.observation_space.spaces[key]
                if tensor.dim() == len(space.shape):
                    tensor = tensor.unsqueeze(0)
                tensors[key] = tensor
            return tensors
        else:
            # Flat array observation
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            tensor = torch.as_tensor(obs).float().to(self.device)
            if tensor.dim() == len(self.observation_space.shape):
                tensor = tensor.unsqueeze(0)
            return tensor

    def _sample_bc_loss(self) -> Optional[torch.Tensor]:
        if self.bc_observations is None or self.bc_actions is None or self.bc_coef <= 0.0:
            return None

        sample_count = min(self.bc_batch_size, len(self.bc_observations))
        indices = np.random.randint(0, len(self.bc_observations), size=sample_count)
        obs_t = self._obs_to_tensor(self.bc_observations[indices])
        action_t = torch.as_tensor(self.bc_actions[indices], dtype=torch.long, device=self.device)
        action_masks = None
        if self.bc_respect_masks and self.bc_action_masks is not None:
            action_masks = self.bc_action_masks[indices]

        dist = self.policy.get_distribution(obs_t, action_masks=action_masks)
        return weighted_multidiscrete_bc_loss(
            dist,
            action_t,
            branch_weights=self.bc_branch_weights,
            nonzero_boosts=self.bc_nonzero_boosts,
            action_value_weights=self.bc_action_value_weights,
        )

    def _sample_advisor_loss(self) -> Optional[torch.Tensor]:
        if self.advisor_coef <= 0.0 or not self.advisor_observations or not self.advisor_actions:
            return None

        sample_count = min(self.advisor_batch_size, len(self.advisor_actions))
        indices = np.random.randint(0, len(self.advisor_actions), size=sample_count)
        obs_batch = np.asarray([self.advisor_observations[index] for index in indices], dtype=np.float32)
        action_batch = np.asarray([self.advisor_actions[index] for index in indices], dtype=np.int64)
        action_masks = None
        if self.advisor_respect_masks and self.advisor_action_masks:
            selected_masks = [self.advisor_action_masks[index] for index in indices]
            if all(mask is not None for mask in selected_masks):
                action_masks = np.asarray(selected_masks, dtype=np.bool_)

        obs_t = self._obs_to_tensor(obs_batch)
        action_t = torch.as_tensor(action_batch, dtype=torch.long, device=self.device)
        dist = self.policy.get_distribution(obs_t, action_masks=action_masks)
        return weighted_multidiscrete_bc_loss(dist, action_t)

    def _remember_advisor_target(
        self,
        obs: Any,
        teacher_action: Optional[List[int]],
        action_mask: Optional[np.ndarray],
    ) -> None:
        if teacher_action is None or self.advisor_coef <= 0.0:
            return
        if isinstance(obs, dict):
            return

        teacher = np.asarray(teacher_action, dtype=np.int64).reshape(-1)
        if teacher.size == 0:
            return
        if action_mask is not None and not self._action_allowed_by_mask(teacher, action_mask):
            return

        self.advisor_observations.append(np.asarray(obs, dtype=np.float32).copy())
        self.advisor_actions.append(teacher.copy())
        self.advisor_action_masks.append(
            np.asarray(action_mask, dtype=np.bool_).copy() if action_mask is not None else None
        )

        if len(self.advisor_actions) > self.advisor_buffer_size:
            overflow = len(self.advisor_actions) - self.advisor_buffer_size
            del self.advisor_observations[:overflow]
            del self.advisor_actions[:overflow]
            del self.advisor_action_masks[:overflow]

    def _action_allowed_by_mask(self, action: np.ndarray, action_mask: np.ndarray) -> bool:
        if action_mask is None:
            return True

        flat_mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
        offset = 0
        for value, dim in zip(action.reshape(-1), self.action_space.nvec):
            value = int(value)
            dim = int(dim)
            if value < 0 or value >= dim:
                return False
            if offset + value >= flat_mask.size or not flat_mask[offset + value]:
                return False
            offset += dim

        return True

    def _extract_boss_hp_percent(self, obs: Any) -> float:
        vector_obs = obs.get("vector") if isinstance(obs, dict) else obs
        if vector_obs is None:
            return float("nan")

        arr = np.asarray(vector_obs, dtype=np.float32).reshape(-1)
        if arr.size <= BOSS_HP_OBS_INDEX:
            return float("nan")

        return float(np.clip(arr[BOSS_HP_OBS_INDEX], 0.0, 1.0) * 100.0)

    def _extract_phase(self, obs: Any) -> float:
        if self.boss_name is None or "lace_boss2_new" not in str(self.boss_name):
            return 1.0

        vector_obs = obs.get("vector") if isinstance(obs, dict) else obs
        if vector_obs is None:
            return float("nan")

        arr = np.asarray(vector_obs, dtype=np.float32).reshape(-1)
        if arr.size < BOSS_PHASE_OBS_OFFSET + BOSS_PHASE_OBS_SIZE:
            return float("nan")

        phase_one_hot = arr[BOSS_PHASE_OBS_OFFSET:BOSS_PHASE_OBS_OFFSET + BOSS_PHASE_OBS_SIZE]
        if not np.any(np.isfinite(phase_one_hot)):
            return float("nan")

        phase = int(np.argmax(phase_one_hot)) + 1
        return float(np.clip(phase, 1, BOSS_PHASE_OBS_SIZE))

    def store_transition(
        self,
        obs: Any,  # Can be List[float] or Dict[str, np.ndarray]
        action: List[int],
        reward: float,
        next_obs: Any,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        reward_components: Optional[Any] = None,
        teacher_action: Optional[List[int]] = None,
    ) -> None:
        """Store a transition in the rollout buffer."""
        # Convert to numpy if flat list
        if isinstance(obs, list):
            obs = np.array(obs, dtype=np.float32)
        if isinstance(next_obs, list):
            next_obs = np.array(next_obs, dtype=np.float32)
        action = np.array(action, dtype=np.int32)

        if not hasattr(self, "current_episode_action_counts"):
            self.current_episode_action_counts = zero_action_counts()
        self.current_episode_action_counts = normalize_action_counts(self.current_episode_action_counts)
        if action.size > 0:
            action_id = int(action.reshape(-1)[0])
            if 0 <= action_id < len(self.current_episode_action_counts):
                self.current_episode_action_counts[action_id] += 1
        self._remember_advisor_target(obs, teacher_action, action_mask)
        
        self.current_episode_reward += reward
        if reward_components is None and not self.warned_missing_reward_components:
            print("[CustomPPO] Warning: reward_components missing from transition payload; component logs will stay at 0 until the Unity mod DLL is updated.")
            self.warned_missing_reward_components = True
        transition_components = normalize_reward_components(reward_components)
        add_reward_components(self.current_episode_components, transition_components)
        if done:

            episode_reward = self.current_episode_reward
            boss_hp_percent = self._extract_boss_hp_percent(next_obs)
            episode_phase = self._extract_phase(next_obs)
            episode_components = dict(self.current_episode_components)
            episode_action_counts = normalize_action_counts(self.current_episode_action_counts)

            self.episodes_completed += 1
            self.episode_rewards.append(episode_reward)
            self.episode_wins.append(reward > 0)
            self.episode_boss_hp_percent.append(boss_hp_percent)
            self.episode_phases.append(episode_phase)
            self.episode_reward_components.append(episode_components)
            self.episode_action_counts.append(episode_action_counts)
            self.current_episode_reward = 0.0
            self.current_episode_components = zero_reward_components()
            self.current_episode_action_counts = zero_action_counts()

            print(
                f"Episode {self.episodes_completed} reward: {episode_reward:.2f}, "
                f"boss remaining HP: {boss_hp_percent:.1f}%, "
                f"phase: {episode_phase:.1f}, "
                f"components: {format_reward_components(episode_components)}"
            )

            rollback_to_best = False
            if self.episodes_completed % WIN_RATE_WINDOW == 0:
                recent_wins = self.episode_wins[-WIN_RATE_WINDOW:]
                recent_rewards = self.episode_rewards[-WIN_RATE_WINDOW:]
                recent_boss_hp = self.episode_boss_hp_percent[-WIN_RATE_WINDOW:]
                recent_phases = self.episode_phases[-WIN_RATE_WINDOW:]
                recent_components = self.episode_reward_components[-WIN_RATE_WINDOW:]
                recent_action_counts = self.episode_action_counts[-WIN_RATE_WINDOW:]
                wins = sum(1 for won in recent_wins if won)
                win_rate = wins / len(recent_wins) if recent_wins else 0.0
                avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                valid_boss_hp = [hp for hp in recent_boss_hp if not np.isnan(hp)]
                avg_boss_hp = float(np.mean(valid_boss_hp)) if valid_boss_hp else float("nan")
                valid_phases = [phase for phase in recent_phases if not np.isnan(phase)]
                avg_phase = float(np.mean(valid_phases)) if valid_phases else 1.0
                checkpoint_score = self._best_checkpoint_score(avg_boss_hp, avg_reward, avg_phase)
                avg_components = average_reward_components(recent_components)
                avg_action_counts = average_action_counts(recent_action_counts)
                print(
                    f"Recent {len(recent_wins)} episode stats after "
                    f"{self.episodes_completed} episodes: {wins}/{len(recent_wins)} "
                    f"wins ({win_rate * 100:.1f}%), avg reward {avg_reward:.2f}, "
                    f"avg boss remaining HP {avg_boss_hp:.1f}%, "
                    f"avg phase {avg_phase:.2f}, checkpoint score {checkpoint_score:.2f}, "
                    f"avg components: {format_reward_components(avg_components)}"
                )
                print(f"Recent action distribution: {format_action_distribution(avg_action_counts)}")
                self._record_window_boss_hp(avg_boss_hp, avg_reward, avg_phase, checkpoint_score, win_rate)
                self._maybe_save_best_checkpoint(avg_boss_hp, avg_reward, avg_phase)
                rollback_to_best = self._maybe_rollback_to_best_checkpoint(avg_boss_hp)

            if rollback_to_best:
                self.last_done = True
                return
            
            if self.save_freq and self.episodes_completed % self.save_freq == 0:
                save_dir = self._boss_directory()
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "checkpoint")
                self.save(save_path)
                # self.plot_rewards(save_dir)
                print(f"Checkpoint saved after {self.episodes_completed} episodes")
        
        obs_t = self._obs_to_tensor(obs)
        action_t = torch.as_tensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.policy.get_distribution(obs_t, action_masks=action_mask)
            value = self.policy.predict_values(obs_t)
            log_prob = dist.log_prob(action_t)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(-1)

        self.rollout_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            episode_start=self.last_done,
            value=value.squeeze(),
            log_prob=log_prob.squeeze(),
            action_masks=action_mask,
        )
        
        self.last_done = done
        
        if self.rollout_buffer.pos >= self.n_steps:
            self.finish_rollout_and_train(next_obs)


    def _record_window_boss_hp(
        self,
        avg_boss_hp: float,
        avg_reward: float,
        avg_phase: float,
        checkpoint_score: float,
        win_rate: float,
    ) -> None:
        if np.isnan(avg_boss_hp):
            return

        self.window_avg_boss_hp_history.append(
            {
                "episode": int(self.episodes_completed),
                "avg_boss_hp": float(avg_boss_hp),
                "avg_reward": float(avg_reward),
                "avg_phase": float(avg_phase),
                "checkpoint_score": float(checkpoint_score),
                "win_rate": float(win_rate),
            }
        )


    def _best_checkpoint_score(self, avg_boss_hp: float, avg_reward: float, avg_phase: float) -> float:
        if np.isnan(avg_boss_hp) or np.isnan(avg_reward) or np.isnan(avg_phase):
            return float("-inf")
        return float(avg_reward) - float(avg_boss_hp) + float(avg_phase) * BEST_CHECKPOINT_PHASE_WEIGHT


    def _maybe_save_best_checkpoint(self, avg_boss_hp: float, avg_reward: float, avg_phase: float) -> None:
        if np.isnan(avg_boss_hp) or np.isnan(avg_reward) or np.isnan(avg_phase):
            return

        checkpoint_score = self._best_checkpoint_score(avg_boss_hp, avg_reward, avg_phase)
        save_reason = None
        if not np.isfinite(self.best_checkpoint_score):
            save_reason = "first valid window"
        else:
            score_improvement = float(checkpoint_score) - float(self.best_checkpoint_score)
            if score_improvement > 0.0:
                save_reason = (
                    f"checkpoint score improved by {score_improvement:.2f} "
                    f"({checkpoint_score:.2f} > {self.best_checkpoint_score:.2f})"
                )

        if save_reason is None:
            return

        self.best_checkpoint_score = float(checkpoint_score)
        self.best_checkpoint_score_metric = BEST_CHECKPOINT_SCORE_METRIC
        self.best_avg_boss_hp_percent = float(avg_boss_hp)
        self.best_checkpoint_avg_reward = float(avg_reward)
        self.best_checkpoint_avg_phase = float(avg_phase)
        self.best_avg_boss_hp_episode = int(self.episodes_completed)
        save_dir = self._boss_directory()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_checkpoint")
        self.save(save_path)
        print(
            f"Best checkpoint saved after {self.episodes_completed} episodes: "
            f"{save_reason}; avg boss remaining HP {avg_boss_hp:.1f}%, "
            f"avg reward {avg_reward:.2f}, avg phase {avg_phase:.2f}, "
            f"checkpoint score {checkpoint_score:.2f}"
        )


    def _maybe_rollback_to_best_checkpoint(self, avg_boss_hp: float) -> bool:
        if np.isnan(avg_boss_hp) or WIN_RATE_WINDOW <= 0:
            return False

        if not np.isfinite(self.best_avg_boss_hp_percent):
            return False

        if len(self.window_avg_boss_hp_history) < ROLLBACK_BAD_WINDOW_COUNT:
            return False

        recent_windows = self.window_avg_boss_hp_history[-ROLLBACK_BAD_WINDOW_COUNT:]
        recent_avg_boss_hp = []
        for window in recent_windows:
            try:
                window_avg_boss_hp = float(window.get("avg_boss_hp", float("nan")))
            except (TypeError, ValueError):
                window_avg_boss_hp = float("nan")
            if np.isnan(window_avg_boss_hp):
                return False
            recent_avg_boss_hp.append(window_avg_boss_hp)

        if any(window.get("episode", 0) <= self.best_avg_boss_hp_episode for window in recent_windows):
            return False

        rollback_threshold = self.best_avg_boss_hp_percent + ROLLBACK_BOSS_HP_TOLERANCE
        if any(window_avg_boss_hp <= rollback_threshold for window_avg_boss_hp in recent_avg_boss_hp):
            return False

        if not self._load_best_checkpoint_parameters():
            return False

        save_dir = self._boss_directory()
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, "checkpoint")
        self.window_avg_boss_hp_history = []
        self.save(checkpoint_path)
        print(
            f"Rolled back to best checkpoint after {self.episodes_completed} episodes: "
            f"last {ROLLBACK_BAD_WINDOW_COUNT} x {WIN_RATE_WINDOW}-episode avg boss remaining HP "
            f"{[round(value, 1) for value in recent_avg_boss_hp]} all > "
            f"best {self.best_avg_boss_hp_percent:.1f}% "
            f"+ tolerance {ROLLBACK_BOSS_HP_TOLERANCE:.1f}%"
        )
        return True


    def _load_best_checkpoint_parameters(self) -> bool:
        best_path = os.path.join(self._boss_directory(), "best_checkpoint.zip")
        if not os.path.exists(best_path):
            print("[CustomPPO] Rollback skipped: best_checkpoint.zip not found")
            return False

        try:
            _, params, _ = load_from_zip_file(best_path, device=self.device)
            self.set_parameters(params, exact_match=False)
            if hasattr(self, "rollout_buffer") and self.rollout_buffer is not None:
                self.rollout_buffer.reset()
            return True
        except Exception as exc:
            print(f"[CustomPPO] Rollback skipped: failed to load best checkpoint: {exc}")
            return False


    def finish_rollout_and_train(self, next_obs: Any) -> None:
        """Finish a rollout and train the model."""
        with torch.no_grad():
            next_obs_t = self._obs_to_tensor(next_obs)
            last_values = self.policy.predict_values(next_obs_t)

        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=np.array([self.last_done])
        )

        self.train()
        self.rollout_buffer.reset()
        self.times_trained += 1

    def train(self) -> None:
        """Update policy using PPO plus an optional decaying BC auxiliary loss."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        bc_losses = []
        advisor_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                bc_loss = self._sample_bc_loss()
                if bc_loss is not None:
                    loss = loss + self.bc_coef * bc_loss
                    bc_losses.append(float(bc_loss.detach().cpu().item()))

                advisor_loss = self._sample_advisor_loss()
                if advisor_loss is not None:
                    loss = loss + self.advisor_coef * advisor_loss
                    advisor_losses.append(float(advisor_loss.detach().cpu().item()))

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if bc_losses:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_coef", self.bc_coef)
            self.bc_coef *= self.bc_coef_decay
        if advisor_losses:
            self.logger.record("train/advisor_loss", np.mean(advisor_losses))
            self.logger.record("train/advisor_coef", self.advisor_coef)
