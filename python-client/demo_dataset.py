import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


DEFAULT_DEMO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos")
MACRO_ACTION_SPACE_SHAPE = [11]
COMPOSITE_ACTION_SPACE_SHAPE = [26]
LOW_LEVEL_ACTION_SPACE_SHAPE = [30]
COMPOSITE_MOVEMENT_COUNT = 6
LEGACY_MACRO_ACTION_SPACE_SHAPES = ([13], [16], [30])
RAW_ACTION_SPACE_SHAPE = [3, 3, 2, 2, 2, 2, 2, 4]

OPTION_KEEP_SWEET_SPOT = 0
OPTION_RETREAT_FROM_BOSS_DODGE = 1
OPTION_DODGE_OVER_BOSS_HIGH_JUMP = 2
OPTION_CROSS_UNDER_AIR_BOSS = 3
OPTION_MOVE_TO_CENTER_SAFELY = 4
OPTION_HOLD_SAFE_SIDE = 5
OPTION_PUNISH_WITH_SLASH = 6
OPTION_ANTI_AIR_SLASH = 7
OPTION_USE_CLOSE_SKILL = 8
OPTION_USE_RANGED_TOOL = 9
OPTION_BIND_HEAL = 10
OPTION11_TO_COMPOSITE26 = {
    OPTION_KEEP_SWEET_SPOT: 0,
    OPTION_RETREAT_FROM_BOSS_DODGE: 1,
    OPTION_DODGE_OVER_BOSS_HIGH_JUMP: 2,
    OPTION_CROSS_UNDER_AIR_BOSS: 3,
    OPTION_MOVE_TO_CENTER_SAFELY: 4,
    OPTION_HOLD_SAFE_SIDE: 5,
    OPTION_PUNISH_WITH_SLASH: 11,
    OPTION_ANTI_AIR_SLASH: 17,
    OPTION_USE_CLOSE_SKILL: 24,
    OPTION_USE_RANGED_TOOL: 23,
    OPTION_BIND_HEAL: 25,
}

MOVE_NONE = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
LOOK_NONE = 0
LOOK_UP = 1
LOOK_DOWN = 2
TOOL_NONE = 0
TOOL_NEUTRAL = 1
TOOL_UP = 2
TOOL_DOWN = 3

RAW_BRANCH_OFFSETS = [0, 3, 6, 8, 10, 12, 14, 16]

LEGACY_CORE_OBS_SIZE = 22
CORE_OBS_SIZE = 24
RESOURCE_OBS_SIZE = 19
LEGACY_SCENE_OBS_SIZE = 10
RAYCAST_COUNT = 32
RAYCAST_HIT_TYPE_COUNT = 6
RAYCAST_OBS_SIZE = RAYCAST_COUNT + RAYCAST_COUNT * RAYCAST_HIT_TYPE_COUNT
HITBOX_OBS_COUNT = 12
HITBOX_TYPE_COUNT = 6
HITBOX_FEATURE_SIZE = 1 + 4 + HITBOX_TYPE_COUNT
HITBOX_OBS_SIZE = HITBOX_OBS_COUNT * HITBOX_FEATURE_SIZE
PRE_RAYCAST_BASE_OBS_SIZE = CORE_OBS_SIZE + RESOURCE_OBS_SIZE + LEGACY_SCENE_OBS_SIZE
LEGACY_BASE_OBS_SIZE = LEGACY_CORE_OBS_SIZE + RESOURCE_OBS_SIZE + LEGACY_SCENE_OBS_SIZE
PRE_HITBOX_BASE_OBS_SIZE = CORE_OBS_SIZE + RESOURCE_OBS_SIZE + RAYCAST_OBS_SIZE
BASE_OBS_SIZE = PRE_HITBOX_BASE_OBS_SIZE + HITBOX_OBS_SIZE
LACE_LEGACY_FSM_ONE_HOT_SIZES = (256, 128, 64)
LACE_COMPRESSED_FSM_ONE_HOT_SIZE = 24
LACE_COMPRESSED_FSM_SEMANTIC_SIZE = 16
LACE_COMPRESSED_FSM_ELAPSED_SIZE = 1
LACE_PHASE_SIZE = 3
LACE_COMPRESSED_FSM_SIZE = (
    LACE_COMPRESSED_FSM_ONE_HOT_SIZE
    + LACE_COMPRESSED_FSM_SEMANTIC_SIZE
    + LACE_COMPRESSED_FSM_ELAPSED_SIZE
)
LACE_COMPRESSED_SINGLE_FRAME_SIZE = BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE + LACE_PHASE_SIZE
LACE_PRE_PHASE_COMPRESSED_SINGLE_FRAME_SIZE = BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE
LACE_PRE_HITBOX_COMPRESSED_SINGLE_FRAME_SIZE = PRE_HITBOX_BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE + LACE_PHASE_SIZE
LACE_PRE_HITBOX_PRE_PHASE_COMPRESSED_SINGLE_FRAME_SIZE = PRE_HITBOX_BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE
LACE_PRE_RAYCAST_COMPRESSED_SINGLE_FRAME_SIZE = PRE_RAYCAST_BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE + LACE_PHASE_SIZE
LACE_PRE_RAYCAST_PRE_PHASE_COMPRESSED_SINGLE_FRAME_SIZE = PRE_RAYCAST_BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE
LACE_LEGACY_COMPRESSED_SINGLE_FRAME_SIZE = LEGACY_BASE_OBS_SIZE + LACE_COMPRESSED_FSM_SIZE
LACE_DENSE_FRAME_STACK_SIZE = 16
LACE_FRAME_SAMPLE_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23)
LACE_COMPRESSED_DENSE_OBS_SIZE = LACE_COMPRESSED_SINGLE_FRAME_SIZE * LACE_DENSE_FRAME_STACK_SIZE
FSM_ELAPSED_NORMALIZATION_SECONDS = 2.0

LACE_PRIORITY_STATES = [
    "Idle",
    "Charge Antic",
    "Charge Break",
    "Charge",
    "Charge Recover",
    "Charge Strike",
    "ComboSlash 1",
    "ComboSlash 3",
    "ComboSlash 5",
    "ComboSlash 7",
    "Combo Strike",
    "Combo Strike Finisher",
    "J Slash M Antic",
    "J Slash Multi",
    "Downstab Antic",
    "Downstab",
    "Downstab Land",
    "RapidSlashAir Antic",
    "RapidSlash Air",
    "RapidSlash Charge",
    "RapidSlash Loop",
    "RapidSlash End",
    "CrossSlash Antic",
    "CrossSlash",
    "Bounce Back",
    "Land",
    "Counter Antic",
    "Counter Hit",
    "Counter Stance",
    "Counter TeleIn",
    "Counter TeleOut",
    "Crossup Antic",
    "ComboSlash 2",
    "ComboSlash 4",
    "ComboSlash 6",
    "Evade",
    "Evade Recover",
    "Hop",
    "Hop Antic",
    "Hop Cancel",
    "Hop Recover",
    "J Slash Multihit",
    "Multihit Slash",
    "Multihitting",
    "P2 Shift 1",
    "P2 Shift 2",
    "P3 Roar",
    "P3 RoarAntic",
    "Pose Swish",
    "Pose Swish 2",
    "Slash End",
    "Slash Slam",
    "Stun Air",
    "Stun Recover",
    "Stunned",
    "Tele In",
    "Tele Out",
    "B Slash 1",
    "B Slash 2",
    "B Slash 3",
    "B Slash End",
    "MultiCharge",
    "MultiCharge Antic",
]

LACE_LEGACY_PRIORITY_STATES_64 = [
    "Bounce Back",
    "Charge",
    "Charge Antic",
    "Charge Break",
    "Charge Recover",
    "Charge Strike",
    "Combo Strike",
    "Combo Strike Finisher",
    "ComboSlash 1",
    "ComboSlash 2",
    "ComboSlash 3",
    "ComboSlash 4",
    "ComboSlash 5",
    "ComboSlash 6",
    "ComboSlash 7",
    "Counter Antic",
    "Counter Hit",
    "Counter Stance",
    "Counter TeleIn",
    "Counter TeleOut",
    "CrossSlash",
    "CrossSlash Antic",
    "Crossup Antic",
    "Downstab",
    "Downstab Antic",
    "Downstab Land",
    "Evade",
    "Evade Recover",
    "Hop",
    "Hop Antic",
    "Hop Cancel",
    "Hop Recover",
    "Idle",
    "J Slash M Antic",
    "J Slash Multi",
    "J Slash Multihit",
    "Land",
    "Multihit Slash",
    "Multihitting",
    "P2 Shift 1",
    "P2 Shift 2",
    "P3 Roar",
    "P3 RoarAntic",
    "Pose Swish",
    "Pose Swish 2",
    "RapidSlash Air",
    "RapidSlash Charge",
    "RapidSlash End",
    "RapidSlash Loop",
    "RapidSlashAir Antic",
    "Slash End",
    "Slash Slam",
    "Stun Air",
    "Stun Recover",
    "Stunned",
    "Tele In",
    "Tele Out",
    "B Slash 1",
    "B Slash 2",
    "B Slash 3",
    "B Slash End",
    "MultiCharge",
    "MultiCharge Antic",
]

LACE_PRIORITY_STATE_TO_INDEX = {name.lower(): index for index, name in enumerate(LACE_PRIORITY_STATES)}

LACE_CONTROL_STATES = sorted([
    "Arena L?",
    "Arena R?",
    "B Slash 1",
    "B Slash 1 Wall",
    "B Slash 2",
    "B Slash 2 Wall",
    "B Slash 3",
    "B Slash End",
    "B Slash Hop",
    "B Slash Hop 2",
    "B Slash HopAntic",
    "Bounce Back",
    "Bounce Back Multihit",
    "Charge",
    "Charge Antic",
    "Charge Break",
    "Charge Crossup?",
    "Charge Recover",
    "Charge Strike",
    "Close",
    "Collide Cancel",
    "Collide To Multihit",
    "Combo Strike",
    "Combo Strike 1",
    "Combo Strike 2",
    "Combo Strike Finisher",
    "ComboSlash 1",
    "ComboSlash 2",
    "ComboSlash 3",
    "ComboSlash 4",
    "ComboSlash 5",
    "ComboSlash 6",
    "ComboSlash 7",
    "Counter Antic",
    "Counter End",
    "Counter Hit",
    "Counter Stance",
    "Counter TeleIn",
    "Counter TeleOut",
    "Counter Type",
    "Cross To J Slash?",
    "CrossSlash",
    "CrossSlash Aim",
    "CrossSlash Aim R",
    "CrossSlash Antic",
    "CrossSlash?",
    "Crossup Antic",
    "CS Evade",
    "CS Evade Cancel",
    "Damage Recover",
    "Death",
    "Death Pose",
    "Distance Check",
    "Downstab",
    "Downstab Antic",
    "Downstab Land",
    "Dstab Angle",
    "Dstab Multihit",
    "Dstab Strike",
    "Evade",
    "Evade Move",
    "Evade Recover",
    "Fall",
    "Far",
    "Finish Multihit",
    "Force CS L?",
    "Force CS R?",
    "Force J Slash",
    "Force L?",
    "Force R?",
    "Hero Facing",
    "Hero Facing 2",
    "Hop",
    "Hop 2",
    "Hop Antic",
    "Hop Antic 2",
    "Hop Cancel",
    "Hop Cancel 2",
    "Hop Check",
    "Hop End",
    "Hop Off Hero?",
    "Hop Recover",
    "Hop Recover 2",
    "Hop To B Slash",
    "Hop To Charge",
    "Hop To Combo",
    "Hop To J Slash",
    "Hop To P2",
    "Hop To P3",
    "Idle",
    "Init",
    "J Slash Antic",
    "J Slash M Antic",
    "J Slash Multi",
    "J Slash Multihit",
    "J Slash Petals?",
    "J Slash Strike",
    "Land",
    "MultiCharge",
    "MultiCharge Antic",
    "MultiCharge Break",
    "MultiCharge Recover",
    "Multihit Slash",
    "Multihit Slash 2",
    "Multihit Slash End",
    "Multihitting",
    "Multihitting 2",
    "P2 Check",
    "P2 Restore Recoil",
    "P2 Shift 1",
    "P2 Shift 2",
    "P3 Check",
    "P3 Idle",
    "P3 Restore Recoil",
    "P3 Roar",
    "P3 RoarAntic",
    "Pause",
    "Pose",
    "Pose Lean",
    "Pose Swish",
    "Pose Swish 2",
    "Pose Upright",
    "Quick Slash 1",
    "Quick Slash 2",
    "Quick Slash 3",
    "RapidSlash Air",
    "RapidSlash Charge",
    "RapidSlash End",
    "RapidSlash Loop",
    "RapidSlashAir Antic",
    "Refight Engarde",
    "Refight Lie",
    "Refight Wake",
    "Repeat?",
    "Reset Slashes",
    "Set CrossSlash Pos",
    "Sing",
    "Sing Antic",
    "Sing End",
    "Slash End",
    "Slash Slam",
    "Start Battle",
    "Start Battle Refight",
    "Start Battle Wait",
    "Start Pause",
    "Steam Damage",
    "Stun Air",
    "Stun Constrain?",
    "Stun Damage",
    "Stun Land",
    "Stun Recover",
    "Stun Start",
    "Stunned",
    "Tele End",
    "Tele In",
    "Tele Out",
    "Trap Stun",
    "Wallcling",
    "Will Counter?",
], key=str.lower)


@dataclass
class DemonstrationDataset:
    observations: np.ndarray
    actions: np.ndarray
    action_masks: Optional[np.ndarray]
    files: List[str]
    boss_name: str


def _normalize(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _action_allowed(action: List[int], mask: List[int], action_shape: List[int]) -> bool:
    joint_size = _joint_action_size(action_shape)
    if len(mask) == joint_size and len(action) == len(action_shape):
        return mask[_flatten_action(action, action_shape)] != 0

    offset = 0
    for value, dim in zip(action, action_shape):
        if value < 0 or value >= dim:
            return False
        if mask[offset + value] == 0:
            return False
        offset += dim
    return True


def _joint_action_size(action_shape: Optional[List[int]]) -> int:
    size = 1
    for dim in list(action_shape or []):
        size *= max(1, int(dim))
    return size


def _flatten_action(action: List[int], action_shape: List[int]) -> int:
    index = 0
    stride = 1
    for value, dim in zip(reversed(action), reversed(action_shape)):
        index += int(value) * stride
        stride *= int(dim)
    return int(index)


def _is_macro_action_space(action_shape: Optional[List[int]]) -> bool:
    return list(action_shape or []) == MACRO_ACTION_SPACE_SHAPE


def _is_composite_action_space(action_shape: Optional[List[int]]) -> bool:
    return list(action_shape or []) == COMPOSITE_ACTION_SPACE_SHAPE


def _is_low_level_action_space(action_shape: Optional[List[int]]) -> bool:
    return list(action_shape or []) == LOW_LEVEL_ACTION_SPACE_SHAPE


def _is_raw_action_space(action_shape: Optional[List[int]]) -> bool:
    return list(action_shape or []) == RAW_ACTION_SPACE_SHAPE


def _is_option13_record(record: Optional[dict]) -> bool:
    if not isinstance(record, dict):
        return False

    action_space_shape = record.get("action_space_shape")
    if action_space_shape == [13]:
        return True

    if action_space_shape == MACRO_ACTION_SPACE_SHAPE:
        return False

    if action_space_shape is not None:
        return False

    action = record.get("action")
    if isinstance(action, list) and len(action) == 1 and isinstance(action[0], int) and 0 <= action[0] < 13:
        mask = record.get("action_mask")
        if isinstance(mask, list) and len(mask) == 13:
            return True

    try:
        return int(record.get("version", 0)) >= 6
    except (TypeError, ValueError):
        return False


def _is_option11_record(record: Optional[dict]) -> bool:
    return isinstance(record, dict) and record.get("action_space_shape") == MACRO_ACTION_SPACE_SHAPE


def _adapt_action(
    action: List[int],
    obs: List[float],
    action_shape: Optional[List[int]],
    record: Optional[dict] = None,
) -> Optional[List[int]]:
    if action_shape is None:
        return action

    if _is_raw_action_space(action_shape):
        if len(action) == len(RAW_ACTION_SPACE_SHAPE):
            return action if all(0 <= value < dim for value, dim in zip(action, RAW_ACTION_SPACE_SHAPE)) else None

        raw_action = record.get("raw_action") if isinstance(record, dict) else None
        if isinstance(raw_action, list) and len(raw_action) == len(RAW_ACTION_SPACE_SHAPE):
            raw_action = [int(value) for value in raw_action]
            return raw_action if all(0 <= value < dim for value, dim in zip(raw_action, RAW_ACTION_SPACE_SHAPE)) else None

        if len(action) == 1:
            value = int(action[0])
            record_shape = record.get("action_space_shape") if isinstance(record, dict) else None
            if record_shape == RAW_ACTION_SPACE_SHAPE:
                return None
            if record_shape == LOW_LEVEL_ACTION_SPACE_SHAPE and 0 <= value < LOW_LEVEL_ACTION_SPACE_SHAPE[0]:
                return _lowlevel_to_raw_action(value, obs)
            if record_shape == COMPOSITE_ACTION_SPACE_SHAPE and 0 <= value < COMPOSITE_ACTION_SPACE_SHAPE[0]:
                return _composite26_to_raw_action(value, obs)
            if record_shape == MACRO_ACTION_SPACE_SHAPE and 0 <= value < MACRO_ACTION_SPACE_SHAPE[0]:
                return _option_to_raw_action(value, obs)
            if _is_option13_record(record):
                return _option_to_raw_action(_option13_to_option11(value), obs)
            if 0 <= value < MACRO_ACTION_SPACE_SHAPE[0]:
                return _option_to_raw_action(value, obs)
            if 0 <= value < COMPOSITE_ACTION_SPACE_SHAPE[0]:
                return _composite26_to_raw_action(value, obs)
            if 0 <= value < LOW_LEVEL_ACTION_SPACE_SHAPE[0]:
                return _lowlevel_to_raw_action(value, obs)

        return None

    if len(action) == len(action_shape):
        if _is_low_level_action_space(action_shape) and len(action) == 1:
            value = int(action[0])
            record_shape = record.get("action_space_shape") if isinstance(record, dict) else None
            raw_action = record.get("raw_action") if isinstance(record, dict) else None
            if record_shape == LOW_LEVEL_ACTION_SPACE_SHAPE and 0 <= value < LOW_LEVEL_ACTION_SPACE_SHAPE[0]:
                return [value]
            if isinstance(raw_action, list) and len(raw_action) == len(RAW_ACTION_SPACE_SHAPE):
                return [_raw_action_to_lowlevel(raw_action, obs)]
            if record_shape == COMPOSITE_ACTION_SPACE_SHAPE and 0 <= value < COMPOSITE_ACTION_SPACE_SHAPE[0]:
                return [_raw_action_to_lowlevel(_composite26_to_raw_action(value, obs), obs)]
            if record_shape == MACRO_ACTION_SPACE_SHAPE and 0 <= value < MACRO_ACTION_SPACE_SHAPE[0]:
                return [_raw_action_to_lowlevel(_option_to_raw_action(value, obs), obs)]
            if _is_option13_record(record):
                return [_raw_action_to_lowlevel(_option_to_raw_action(_option13_to_option11(value), obs), obs)]
            if 0 <= value < MACRO_ACTION_SPACE_SHAPE[0]:
                return [_raw_action_to_lowlevel(_option_to_raw_action(value, obs), obs)]
            if 0 <= value < COMPOSITE_ACTION_SPACE_SHAPE[0]:
                return [_raw_action_to_lowlevel(_composite26_to_raw_action(value, obs), obs)]

        if _is_composite_action_space(action_shape) and len(action) == 1:
            value = int(action[0])
            record_shape = record.get("action_space_shape") if isinstance(record, dict) else None
            if record_shape == COMPOSITE_ACTION_SPACE_SHAPE and 0 <= value < COMPOSITE_ACTION_SPACE_SHAPE[0]:
                return [value]
            if record_shape == MACRO_ACTION_SPACE_SHAPE and 0 <= value < MACRO_ACTION_SPACE_SHAPE[0]:
                return [OPTION11_TO_COMPOSITE26.get(value, 5)]
            if _is_option13_record(record):
                return [OPTION11_TO_COMPOSITE26.get(_option13_to_option11(value), 5)]
            if 0 <= value < MACRO_ACTION_SPACE_SHAPE[0]:
                return [OPTION11_TO_COMPOSITE26.get(value, 5)]
            if 0 <= value < COMPOSITE_ACTION_SPACE_SHAPE[0]:
                return [value]

        if _is_macro_action_space(action_shape) and len(action) == 1 and _is_option13_record(record):
            return [_option13_to_option11(int(action[0]))]

        if _is_macro_action_space(action_shape) and len(action) == 1 and _is_option11_record(record):
            return action if all(0 <= value < dim for value, dim in zip(action, action_shape)) else None

        if all(0 <= value < dim for value, dim in zip(action, action_shape)):
            return action

        if _is_macro_action_space(action_shape) and len(action) == 1:
            return [_legacy_macro_to_option(int(action[0]), obs)]

        if _is_composite_action_space(action_shape) and len(action) == 1:
            return [OPTION11_TO_COMPOSITE26.get(_legacy_macro_to_option(int(action[0]), obs), 5)]

    if _is_macro_action_space(action_shape) and len(action) == len(RAW_ACTION_SPACE_SHAPE):
        return [_raw_action_to_option(action, obs)]

    if _is_composite_action_space(action_shape) and len(action) == len(RAW_ACTION_SPACE_SHAPE):
        return [_raw_action_to_composite26(action, obs)]

    if _is_low_level_action_space(action_shape) and len(action) == len(RAW_ACTION_SPACE_SHAPE):
        return [_raw_action_to_lowlevel(action, obs)]

    return None


def _adapt_mask(
    mask: Optional[List[int]],
    obs: List[float],
    action_shape: Optional[List[int]],
    record: Optional[dict] = None,
) -> Optional[List[int]]:
    if mask is None or action_shape is None:
        return mask

    expected_size = sum(action_shape)
    if len(mask) == expected_size:
        return mask

    if _is_raw_action_space(action_shape):
        if len(mask) == _joint_action_size(RAW_ACTION_SPACE_SHAPE):
            return mask
        if len(mask) == LOW_LEVEL_ACTION_SPACE_SHAPE[0]:
            return _lowlevel_mask_to_raw(mask, obs)
        if len(mask) == COMPOSITE_ACTION_SPACE_SHAPE[0]:
            return _composite26_mask_to_raw(mask, obs)
        if len(mask) == MACRO_ACTION_SPACE_SHAPE[0]:
            return _option11_mask_to_raw(mask, obs)
        if len(mask) == 13:
            return _option11_mask_to_raw(_option13_mask_to_option11(mask), obs)
        if any(len(mask) == sum(shape) for shape in LEGACY_MACRO_ACTION_SPACE_SHAPES):
            return _option11_mask_to_raw(_legacy_macro_mask_to_option(mask), obs)

    if _is_low_level_action_space(action_shape):
        if len(mask) == sum(RAW_ACTION_SPACE_SHAPE):
            return _raw_mask_to_lowlevel(mask, obs)
        if len(mask) == COMPOSITE_ACTION_SPACE_SHAPE[0]:
            return _composite26_mask_to_lowlevel(mask, obs)
        if len(mask) == MACRO_ACTION_SPACE_SHAPE[0]:
            return _option11_mask_to_lowlevel(mask, obs)
        if len(mask) == 13:
            return _option11_mask_to_lowlevel(_option13_mask_to_option11(mask), obs)
        if any(len(mask) == sum(shape) for shape in LEGACY_MACRO_ACTION_SPACE_SHAPES):
            return _option11_mask_to_lowlevel(_legacy_macro_mask_to_option(mask), obs)

    if _is_macro_action_space(action_shape) and len(mask) == 13:
        return _option13_mask_to_option11(mask)

    if _is_composite_action_space(action_shape) and len(mask) == 13:
        return _option11_mask_to_composite26(_option13_mask_to_option11(mask))

    if _is_macro_action_space(action_shape) and any(len(mask) == sum(shape) for shape in LEGACY_MACRO_ACTION_SPACE_SHAPES):
        return _legacy_macro_mask_to_option(mask)

    if _is_composite_action_space(action_shape) and any(len(mask) == sum(shape) for shape in LEGACY_MACRO_ACTION_SPACE_SHAPES):
        return _option11_mask_to_composite26(_legacy_macro_mask_to_option(mask))

    if _is_macro_action_space(action_shape) and len(mask) == sum(RAW_ACTION_SPACE_SHAPE):
        return _raw_mask_to_option(mask, obs)

    if _is_composite_action_space(action_shape) and len(mask) == sum(RAW_ACTION_SPACE_SHAPE):
        return _raw_mask_to_composite26(mask, obs)

    return None


def _adapt_observation_sequence(records: List[dict], target_obs_size: Optional[int]) -> List[Optional[List[float]]]:
    if target_obs_size is None:
        return [record.get("observation") for record in records]

    raw_observations = [record.get("observation") for record in records]
    if all(isinstance(obs, list) and len(obs) == target_obs_size for obs in raw_observations):
        return raw_observations

    if target_obs_size != LACE_COMPRESSED_DENSE_OBS_SIZE:
        return [
            obs if isinstance(obs, list) and len(obs) == target_obs_size else None
            for obs in raw_observations
        ]

    single_frames = _extract_lace_single_frames(records)
    if not any(frame is not None for frame in single_frames):
        return [None] * len(records)

    return _build_dense_lace_observations(single_frames)


def _extract_lace_single_frames(records: List[dict]) -> List[Optional[List[float]]]:
    frames: List[Optional[List[float]]] = []
    previous_state: Optional[int] = None
    state_entered_time = 0.0
    previous_time = 0.0
    phase = 1

    for record in records:
        frame = _extract_lace_current_frame(record.get("observation"))
        current_time = _record_time(record, previous_time)
        previous_time = current_time

        if frame is None:
            frames.append(None)
            continue

        current_state = _lace_fsm_state_id(frame)
        if previous_state is None or current_state != previous_state:
            previous_state = current_state
            state_entered_time = current_time

        elapsed_seconds = max(0.0, current_time - state_entered_time)
        phase = _update_lace_phase(phase, _lace_frame_state_name(frame))
        frames.append(_ensure_lace_elapsed_frame(frame, elapsed_seconds, phase))

    return frames


def _extract_lace_current_frame(obs: object) -> Optional[List[float]]:
    if not isinstance(obs, list):
        return None

    if _is_lace_single_frame_size(len(obs)):
        return obs

    for stack_size in (LACE_DENSE_FRAME_STACK_SIZE, 8, 32):
        if len(obs) % stack_size != 0:
            continue

        frame_size = len(obs) // stack_size
        if _is_lace_single_frame_size(frame_size):
            return obs[:frame_size]

    return None


def _is_lace_single_frame_size(size: int) -> bool:
    if size == LACE_COMPRESSED_SINGLE_FRAME_SIZE:
        return True
    if size == LACE_PRE_PHASE_COMPRESSED_SINGLE_FRAME_SIZE:
        return True
    if size == LACE_PRE_HITBOX_COMPRESSED_SINGLE_FRAME_SIZE:
        return True
    if size == LACE_PRE_HITBOX_PRE_PHASE_COMPRESSED_SINGLE_FRAME_SIZE:
        return True
    if size == LACE_PRE_RAYCAST_COMPRESSED_SINGLE_FRAME_SIZE:
        return True
    if size == LACE_PRE_RAYCAST_PRE_PHASE_COMPRESSED_SINGLE_FRAME_SIZE:
        return True
    if size == LACE_LEGACY_COMPRESSED_SINGLE_FRAME_SIZE:
        return True

    for one_hot_size in LACE_LEGACY_FSM_ONE_HOT_SIZES:
        for base_size in (BASE_OBS_SIZE, PRE_HITBOX_BASE_OBS_SIZE, PRE_RAYCAST_BASE_OBS_SIZE, LEGACY_BASE_OBS_SIZE):
            if size in (base_size + one_hot_size, base_size + one_hot_size + 1):
                return True

    return False


def _record_time(record: dict, fallback_time: float) -> float:
    value = record.get("time")
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback_time + 0.025


def _lace_fsm_state_id(frame: List[float]) -> int:
    one_hot_size = _lace_frame_one_hot_size(frame)
    fsm_start = _lace_frame_base_size(frame)
    fsm_end = fsm_start + one_hot_size
    one_hot = frame[fsm_start:fsm_end]
    if not one_hot:
        return 0
    return int(np.argmax(np.asarray(one_hot, dtype=np.float32)))


def _ensure_lace_elapsed_frame(frame: List[float], elapsed_seconds: float, phase: int = 1) -> List[float]:
    if len(frame) == LACE_COMPRESSED_SINGLE_FRAME_SIZE:
        upgraded = _upgrade_lace_base_frame(frame)
        if upgraded == frame[:BASE_OBS_SIZE]:
            return frame

    return _compress_lace_frame(frame, elapsed_seconds, phase)


def _compress_lace_frame(frame: List[float], elapsed_seconds: float, phase: int = 1) -> List[float]:
    compressed = _upgrade_lace_base_frame(frame)
    _rewrite_resource_features(compressed)

    state_name = _lace_frame_state_name(frame)
    one_hot = [0.0] * LACE_COMPRESSED_FSM_ONE_HOT_SIZE
    state_index = LACE_PRIORITY_STATE_TO_INDEX.get(state_name.lower()) if state_name else None
    if state_index is None or state_index >= LACE_COMPRESSED_FSM_ONE_HOT_SIZE - 1:
        one_hot[0] = 1.0
    else:
        one_hot[1 + state_index] = 1.0

    semantic = _lace_semantic_flags(state_name, compressed)
    elapsed_normalized = _lace_frame_elapsed(frame, elapsed_seconds)
    phase_one_hot = [0.0] * LACE_PHASE_SIZE
    phase_one_hot[min(max(int(phase), 1), 3) - 1] = 1.0
    return compressed + one_hot + semantic + [elapsed_normalized] + phase_one_hot


def _rewrite_resource_features(frame: List[float]) -> None:
    if len(frame) < BASE_OBS_SIZE:
        return

    silk_index = CORE_OBS_SIZE
    silk = frame[silk_index] * 18.0 if len(frame) > silk_index else 0.0
    hero_hp = frame[4] if len(frame) > 4 else 1.0
    injured = hero_hp < 0.999
    frame[silk_index + 2] = 1.0 if injured and silk >= 9.0 else 0.0
    frame[silk_index + 3] = min(1.0, max(0.0, 9.0 - silk) / 9.0)


def _lace_frame_one_hot_size(frame: List[float]) -> int:
    extra = len(frame) - _lace_frame_base_size(frame)
    if extra in (LACE_COMPRESSED_FSM_SIZE, LACE_COMPRESSED_FSM_SIZE + LACE_PHASE_SIZE):
        return LACE_COMPRESSED_FSM_ONE_HOT_SIZE

    for one_hot_size in LACE_LEGACY_FSM_ONE_HOT_SIZES:
        if extra in (one_hot_size, one_hot_size + 1):
            return one_hot_size

    return max(0, extra)


def _lace_frame_base_size(frame: List[float]) -> int:
    size = len(frame)
    for base_size in (BASE_OBS_SIZE, PRE_HITBOX_BASE_OBS_SIZE, PRE_RAYCAST_BASE_OBS_SIZE, LEGACY_BASE_OBS_SIZE):
        extra = size - base_size
        if extra in (LACE_COMPRESSED_FSM_SIZE, LACE_COMPRESSED_FSM_SIZE + LACE_PHASE_SIZE):
            return base_size
        for one_hot_size in LACE_LEGACY_FSM_ONE_HOT_SIZES:
            if extra in (one_hot_size, one_hot_size + 1):
                return base_size

    if size >= BASE_OBS_SIZE:
        return BASE_OBS_SIZE
    if size >= PRE_HITBOX_BASE_OBS_SIZE:
        return PRE_HITBOX_BASE_OBS_SIZE
    if size >= PRE_RAYCAST_BASE_OBS_SIZE:
        return PRE_RAYCAST_BASE_OBS_SIZE
    return LEGACY_BASE_OBS_SIZE


def _upgrade_lace_base_frame(frame: List[float]) -> List[float]:
    base_size = _lace_frame_base_size(frame)
    if base_size == BASE_OBS_SIZE:
        upgraded = list(frame[:BASE_OBS_SIZE])
        if len(upgraded) > 23:
            upgraded[22] = 1.0 if _facing_boss(upgraded) else 0.0
            upgraded[23] = 1.0 if _estimate_boss_facing_right(upgraded) else 0.0
        return upgraded

    if base_size == PRE_HITBOX_BASE_OBS_SIZE:
        upgraded = list(frame[:PRE_HITBOX_BASE_OBS_SIZE])
        if len(upgraded) > 23:
            upgraded[22] = 1.0 if _facing_boss(upgraded) else 0.0
            upgraded[23] = 1.0 if _estimate_boss_facing_right(upgraded) else 0.0
        return upgraded + [0.0] * HITBOX_OBS_SIZE

    if base_size == PRE_RAYCAST_BASE_OBS_SIZE:
        upgraded = list(frame[:CORE_OBS_SIZE + RESOURCE_OBS_SIZE])
        if len(upgraded) > 23:
            upgraded[22] = 1.0 if _facing_boss(upgraded) else 0.0
            upgraded[23] = 1.0 if _estimate_boss_facing_right(upgraded) else 0.0
        return upgraded + [0.0] * RAYCAST_OBS_SIZE + [0.0] * HITBOX_OBS_SIZE

    legacy_base = list(frame[:LEGACY_BASE_OBS_SIZE])
    if len(legacy_base) < LEGACY_BASE_OBS_SIZE:
        return legacy_base + [0.0] * (BASE_OBS_SIZE - len(legacy_base))

    hero_facing_boss = 1.0 if _facing_boss(legacy_base) else 0.0
    boss_facing_right = 1.0 if _estimate_boss_facing_right(legacy_base) else 0.0
    upgraded = (
        legacy_base[:LEGACY_CORE_OBS_SIZE]
        + [hero_facing_boss, boss_facing_right]
        + legacy_base[LEGACY_CORE_OBS_SIZE:LEGACY_CORE_OBS_SIZE + RESOURCE_OBS_SIZE]
    )
    return upgraded + [0.0] * RAYCAST_OBS_SIZE + [0.0] * HITBOX_OBS_SIZE


def _lace_frame_state_name(frame: List[float]) -> str:
    one_hot_size = _lace_frame_one_hot_size(frame)
    if one_hot_size <= 0:
        return ""

    fsm_start = _lace_frame_base_size(frame)
    one_hot = frame[fsm_start:fsm_start + one_hot_size]
    if not one_hot:
        return ""

    slot = int(np.argmax(np.asarray(one_hot, dtype=np.float32)))
    if slot == 0:
        return ""

    if one_hot_size == LACE_COMPRESSED_FSM_ONE_HOT_SIZE:
        state_index = slot - 1
        return LACE_PRIORITY_STATES[state_index] if state_index < len(LACE_PRIORITY_STATES) else ""

    if one_hot_size == 64:
        state_index = slot - 1
        return LACE_LEGACY_PRIORITY_STATES_64[state_index] if state_index < len(LACE_LEGACY_PRIORITY_STATES_64) else ""

    state_index = slot - 1
    return LACE_CONTROL_STATES[state_index] if state_index < len(LACE_CONTROL_STATES) else ""


def _lace_frame_elapsed(frame: List[float], elapsed_seconds: float) -> float:
    elapsed_index = _lace_frame_elapsed_index(frame)
    if len(frame) > elapsed_index:
        try:
            return min(1.0, max(0.0, float(frame[elapsed_index])))
        except (TypeError, ValueError):
            pass

    return min(1.0, max(0.0, elapsed_seconds) / FSM_ELAPSED_NORMALIZATION_SECONDS)


def _lace_frame_elapsed_index(frame: List[float]) -> int:
    base_size = _lace_frame_base_size(frame)
    one_hot_size = _lace_frame_one_hot_size(frame)
    extra = len(frame) - base_size
    if one_hot_size == LACE_COMPRESSED_FSM_ONE_HOT_SIZE and extra in (
        LACE_COMPRESSED_FSM_SIZE,
        LACE_COMPRESSED_FSM_SIZE + LACE_PHASE_SIZE,
    ):
        return base_size + LACE_COMPRESSED_FSM_ONE_HOT_SIZE + LACE_COMPRESSED_FSM_SEMANTIC_SIZE

    return base_size + one_hot_size


def _update_lace_phase(current_phase: int, state_name: str) -> int:
    state = (state_name or "").lower()
    if "p3" in state:
        return 3
    if "p2" in state:
        return max(current_phase, 2)
    return max(1, min(current_phase, 3))


def _contains_any(value: str, *tokens: str) -> bool:
    value = value.lower()
    return any(token in value for token in tokens)


def _boss_attack_opportunity_state(state: str) -> bool:
    return _contains_any(state, "recover", "stun", "stunned", "land", "bounce back", "slash end")


def _boss_safe_for_bind_state(state: str) -> bool:
    return _contains_any(state, "idle", "recover", "stun", "stunned", "land", "slash end")


def _boss_dangerous_state(state: str) -> bool:
    if _boss_attack_opportunity_state(state):
        return False
    return _contains_any(
        state,
        "attack",
        "slash",
        "dash",
        "lunge",
        "stab",
        "throw",
        "sphere",
        "needle",
        "parry",
        "counter",
        "strike",
        "hit",
        "slam",
    )


def _boss_ground_rush_state(state: str) -> bool:
    if _boss_attack_opportunity_state(state):
        return False
    if _contains_any(state, "dash", "lunge", "stab", "charge"):
        return True
    return (
        _contains_any(state, "step", "run", "walk")
        and _contains_any(state, "slash", "attack", "combo")
    ) or _contains_any(state, "slash step", "step slash", "multi slash")


def _lace_semantic_flags(state_name: str, frame: List[float]) -> List[float]:
    state = (state_name or "").lower()
    dangerous = _boss_dangerous_state(state)
    attack_opportunity = _boss_attack_opportunity_state(state)
    safe_for_bind = _boss_safe_for_bind_state(state)
    ground_rush = _boss_ground_rush_state(state)

    rel_x, rel_y, abs_x, abs_y, distance, facing_boss = _lace_geometry(frame)
    boss_above_hero = rel_y > 1.0
    near_collision = distance <= 0.75
    near_danger_reach = dangerous and abs_x <= 4.2 and abs_y <= 2.2
    flat_slash_window = attack_opportunity and facing_boss and 0.9 <= abs_x <= 2.8 and abs_y <= 1.2
    vertical_slash_window = attack_opportunity and 0.5 <= rel_y <= 3.2 and abs_x <= 1.5

    silk = frame[CORE_OBS_SIZE] * 18.0 if len(frame) > CORE_OBS_SIZE else 0.0
    hero_hp = frame[4] if len(frame) > 4 else 1.0
    safe_bind_window = silk >= 9.0 and hero_hp < 0.999 and (safe_for_bind or distance >= 9.0)

    startup = _contains_any(
        state,
        "antic",
        "aim",
        "break",
        "stance",
        "tele in",
        "tele out",
        "roarantic",
        "crossup",
    )
    recovery = _contains_any(state, "recover", "land", "bounce", "slash end", "rapidslash end", "end")
    stun = _contains_any(state, "stun", "stunned")
    counter_or_parry = _contains_any(state, "counter", "parry")
    teleport_or_phase = _contains_any(state, "tele", "shift", "roar", "pose")
    active_damage = (
        dangerous
        and not startup
        and not recovery
        and not stun
        and _contains_any(
            state,
            "slash",
            "strike",
            "hit",
            "downstab",
            "charge",
            "rapid",
            "multi",
            "slam",
            "crossslash",
        )
    )

    return [
        float(dangerous),
        float(attack_opportunity),
        float(safe_for_bind),
        float(ground_rush),
        float(boss_above_hero),
        float(near_collision),
        float(near_danger_reach),
        float(flat_slash_window),
        float(vertical_slash_window),
        float(safe_bind_window),
        float(startup),
        float(active_damage),
        float(recovery),
        float(stun),
        float(counter_or_parry),
        float(teleport_or_phase),
    ]


def _lace_geometry(frame: List[float]) -> tuple[float, float, float, float, float, bool]:
    if not isinstance(frame, list) or len(frame) <= 21:
        return 0.0, 0.0, 999.0, 999.0, 999.0, False

    arena_width = 44.0
    arena_height = 14.0
    hero_x = 33.0 + frame[0] * arena_width
    hero_y = 96.0 + frame[1] * arena_height
    boss_x = 33.0 + frame[5] * arena_width
    boss_y = 96.0 + frame[6] * arena_height
    rel_x = boss_x - hero_x
    rel_y = boss_y - hero_y
    abs_x = abs(rel_x)
    abs_y = abs(rel_y)
    distance = float(np.hypot(rel_x, rel_y))
    boss_is_right = rel_x >= 0.0
    hero_facing_right = frame[21] > 0.5
    facing_boss = hero_facing_right if boss_is_right else not hero_facing_right
    return rel_x, rel_y, abs_x, abs_y, distance, facing_boss


def _build_dense_lace_observations(single_frames: List[Optional[List[float]]]) -> List[Optional[List[float]]]:
    first_valid = next((frame for frame in single_frames if frame is not None), None)
    if first_valid is None:
        return [None] * len(single_frames)

    stacked: List[Optional[List[float]]] = []
    for index in range(len(single_frames)):
        frames: List[List[float]] = []
        for offset in LACE_FRAME_SAMPLE_OFFSETS:
            source_index = index - offset
            frame = single_frames[source_index] if source_index >= 0 else None
            frames.append(frame if frame is not None else first_valid)
        stacked.append([value for frame in frames for value in frame])

    return stacked


def _raw_action_to_option(action: List[int], obs: List[float]) -> int:
    move = action[0]
    look = action[1]
    jump = action[2] == 1
    attack = action[3] == 1
    dash = action[4] == 1
    needle = action[5] == 1
    bind = action[6] == 1
    tool = action[7]

    if bind:
        return OPTION_BIND_HEAL
    if tool == TOOL_NEUTRAL:
        return OPTION_USE_CLOSE_SKILL
    if tool == TOOL_UP or tool == TOOL_DOWN:
        return OPTION_USE_RANGED_TOOL
    if needle:
        return OPTION_USE_RANGED_TOOL
    if attack:
        if look == LOOK_UP:
            return OPTION_ANTI_AIR_SLASH
        return OPTION_PUNISH_WITH_SLASH
    if dash:
        center = _direction_to_center(obs)
        if move != MOVE_NONE and move == center and _far_from_center(obs):
            return OPTION_MOVE_TO_CENTER_SAFELY
        return OPTION_DODGE_OVER_BOSS_HIGH_JUMP if move == _direction_to_boss(obs) else OPTION_RETREAT_FROM_BOSS_DODGE
    if jump:
        center = _direction_to_center(obs)
        if move != MOVE_NONE and move == center and _far_from_center(obs):
            return OPTION_MOVE_TO_CENTER_SAFELY
        toward = _direction_to_boss(obs)
        if move == toward:
            return OPTION_DODGE_OVER_BOSS_HIGH_JUMP
        if move == _opposite_direction(toward):
            return OPTION_RETREAT_FROM_BOSS_DODGE
        return OPTION_HOLD_SAFE_SIDE
    if move != MOVE_NONE:
        center_direction = _direction_to_center(obs)
        if move == center_direction and _far_from_center(obs):
            return OPTION_MOVE_TO_CENTER_SAFELY
        return OPTION_KEEP_SWEET_SPOT if move == _direction_to_boss(obs) else OPTION_RETREAT_FROM_BOSS_DODGE

    return OPTION_HOLD_SAFE_SIDE


def _raw_action_to_movement_option(action: List[int], obs: List[float]) -> int:
    move = action[0]
    jump = action[2] == 1
    dash = action[4] == 1

    if dash:
        center = _direction_to_center(obs)
        if move != MOVE_NONE and move == center and _far_from_center(obs):
            return OPTION_MOVE_TO_CENTER_SAFELY
        return OPTION_DODGE_OVER_BOSS_HIGH_JUMP if move == _direction_to_boss(obs) else OPTION_RETREAT_FROM_BOSS_DODGE
    if jump:
        center = _direction_to_center(obs)
        if move != MOVE_NONE and move == center and _far_from_center(obs):
            return OPTION_MOVE_TO_CENTER_SAFELY
        toward = _direction_to_boss(obs)
        if move == toward:
            return OPTION_DODGE_OVER_BOSS_HIGH_JUMP
        if move == _opposite_direction(toward):
            return OPTION_RETREAT_FROM_BOSS_DODGE
        return OPTION_HOLD_SAFE_SIDE
    if move != MOVE_NONE:
        center_direction = _direction_to_center(obs)
        if move == center_direction and _far_from_center(obs):
            return OPTION_MOVE_TO_CENTER_SAFELY
        return OPTION_KEEP_SWEET_SPOT if move == _direction_to_boss(obs) else OPTION_RETREAT_FROM_BOSS_DODGE

    return OPTION_HOLD_SAFE_SIDE


def _raw_action_to_composite26(action: List[int], obs: List[float]) -> int:
    look = action[1]
    attack = action[3] == 1
    needle = action[5] == 1
    bind = action[6] == 1
    tool = action[7]

    if bind:
        return 25
    if tool == TOOL_NEUTRAL:
        return 24

    movement = _raw_action_to_movement_option(action, obs)
    offense = 0
    if tool == TOOL_UP or tool == TOOL_DOWN or needle:
        offense = 3
    elif attack:
        offense = 2 if look == LOOK_UP else 1

    return offense * COMPOSITE_MOVEMENT_COUNT + movement


def _raw_action_to_lowlevel(action: List[int], obs: List[float]) -> int:
    move = int(action[0]) if len(action) > 0 else MOVE_NONE
    look = int(action[1]) if len(action) > 1 else LOOK_NONE
    jump = len(action) > 2 and int(action[2]) == 1
    attack = len(action) > 3 and int(action[3]) == 1
    dash = len(action) > 4 and int(action[4]) == 1
    needle = len(action) > 5 and int(action[5]) == 1
    bind = len(action) > 6 and int(action[6]) == 1
    tool = int(action[7]) if len(action) > 7 else TOOL_NONE

    if bind:
        return 22
    if tool != TOOL_NONE:
        if tool == TOOL_UP:
            return 24
        if tool == TOOL_DOWN:
            return 25
        if jump and move == MOVE_LEFT:
            return 28
        if jump and move == MOVE_RIGHT:
            return 29
        if move == MOVE_LEFT:
            return 26
        if move == MOVE_RIGHT:
            return 27
        return 23
    if needle:
        return 21
    if attack:
        if dash and move == MOVE_LEFT:
            return 19
        if dash and move == MOVE_RIGHT:
            return 20
        if jump and move == MOVE_LEFT:
            return 17
        if jump and move == MOVE_RIGHT:
            return 18
        if jump:
            return 16
        if look == LOOK_UP:
            return 14
        if look == LOOK_DOWN:
            return 15
        if move == MOVE_LEFT:
            return 12
        if move == MOVE_RIGHT:
            return 13
        return 11
    if dash:
        if move == MOVE_LEFT:
            return 9
        if move == MOVE_RIGHT:
            return 10
        return 8
    if jump:
        if move == MOVE_LEFT:
            return 6
        if move == MOVE_RIGHT:
            return 7
        return 5
    if look == LOOK_UP:
        return 3
    if look == LOOK_DOWN:
        return 4
    if move == MOVE_LEFT:
        return 1
    if move == MOVE_RIGHT:
        return 2
    return 0


def _lowlevel_to_raw_action(action: int, obs: List[float]) -> List[int]:
    raw = [0] * len(RAW_ACTION_SPACE_SHAPE)
    action = int(action)

    if action == 1:
        raw[0] = MOVE_LEFT
    elif action == 2:
        raw[0] = MOVE_RIGHT
    elif action == 3:
        raw[1] = LOOK_UP
    elif action == 4:
        raw[1] = LOOK_DOWN
    elif action == 5:
        raw[2] = 1
    elif action == 6:
        raw[0] = MOVE_LEFT; raw[2] = 1
    elif action == 7:
        raw[0] = MOVE_RIGHT; raw[2] = 1
    elif action == 8:
        raw[4] = 1
    elif action == 9:
        raw[0] = MOVE_LEFT; raw[4] = 1
    elif action == 10:
        raw[0] = MOVE_RIGHT; raw[4] = 1
    elif action == 11:
        raw[3] = 1
        if not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)
    elif action == 12:
        raw[0] = MOVE_LEFT; raw[3] = 1
    elif action == 13:
        raw[0] = MOVE_RIGHT; raw[3] = 1
    elif action == 14:
        raw[1] = LOOK_UP; raw[3] = 1
    elif action == 15:
        raw[1] = LOOK_DOWN; raw[3] = 1
    elif action == 16:
        raw[2] = 1; raw[3] = 1
        if not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)
    elif action == 17:
        raw[0] = MOVE_LEFT; raw[2] = 1; raw[3] = 1
    elif action == 18:
        raw[0] = MOVE_RIGHT; raw[2] = 1; raw[3] = 1
    elif action == 19:
        raw[0] = MOVE_LEFT; raw[3] = 1; raw[4] = 1
    elif action == 20:
        raw[0] = MOVE_RIGHT; raw[3] = 1; raw[4] = 1
    elif action == 21:
        raw[5] = 1
        if not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)
    elif action == 22:
        raw[6] = 1
    elif action == 23:
        raw[7] = TOOL_NEUTRAL
        if not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)
    elif action == 24:
        raw[7] = TOOL_UP
        if not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)
    elif action == 25:
        raw[7] = TOOL_DOWN
        if not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)
    elif action == 26:
        raw[0] = MOVE_LEFT; raw[7] = TOOL_NEUTRAL
    elif action == 27:
        raw[0] = MOVE_RIGHT; raw[7] = TOOL_NEUTRAL
    elif action == 28:
        raw[0] = MOVE_LEFT; raw[2] = 1; raw[7] = TOOL_NEUTRAL
    elif action == 29:
        raw[0] = MOVE_RIGHT; raw[2] = 1; raw[7] = TOOL_NEUTRAL

    return raw


def _legacy_macro_to_option(macro: int, obs: List[float]) -> int:
    if macro in (11, 12):
        return OPTION_BIND_HEAL if macro == 11 else OPTION_USE_RANGED_TOOL
    if macro == 13:
        return OPTION_USE_CLOSE_SKILL
    if macro in (14, 15):
        return OPTION_USE_RANGED_TOOL
    if macro in (9,):
        return OPTION_ANTI_AIR_SLASH
    if macro in (8, 10, 26, 27, 28, 29):
        return OPTION_PUNISH_WITH_SLASH
    if macro in (5, 7, 21):
        return OPTION_DODGE_OVER_BOSS_HIGH_JUMP
    if macro in (6, 18, 22):
        return OPTION_RETREAT_FROM_BOSS_DODGE
    if macro in (4, 19, 23, 24):
        return OPTION_MOVE_TO_CENTER_SAFELY
    if macro in (1, 3, 17):
        return OPTION_KEEP_SWEET_SPOT
    if macro in (2,):
        return OPTION_RETREAT_FROM_BOSS_DODGE
    if macro in (20, 16, 0):
        return OPTION_HOLD_SAFE_SIDE
    if macro in (25,):
        return OPTION_DODGE_OVER_BOSS_HIGH_JUMP

    return OPTION_HOLD_SAFE_SIDE


def _option13_to_option11(option: int) -> int:
    mapping = {
        0: OPTION_KEEP_SWEET_SPOT,
        1: OPTION_RETREAT_FROM_BOSS_DODGE,
        2: OPTION_DODGE_OVER_BOSS_HIGH_JUMP,
        3: OPTION_CROSS_UNDER_AIR_BOSS,
        4: OPTION_MOVE_TO_CENTER_SAFELY,
        5: OPTION_KEEP_SWEET_SPOT,
        6: OPTION_HOLD_SAFE_SIDE,
        7: OPTION_PUNISH_WITH_SLASH,
        8: OPTION_ANTI_AIR_SLASH,
        9: OPTION_USE_CLOSE_SKILL,
        10: OPTION_USE_RANGED_TOOL,
        11: OPTION_RETREAT_FROM_BOSS_DODGE,
        12: OPTION_BIND_HEAL,
    }
    return mapping.get(option, OPTION_HOLD_SAFE_SIDE)


def _option13_mask_to_option11(mask: List[int]) -> List[int]:
    option_mask = [0] * MACRO_ACTION_SPACE_SHAPE[0]
    for option, enabled in enumerate(mask):
        if enabled:
            option_mask[_option13_to_option11(option)] = 1

    if not any(option_mask):
        option_mask[OPTION_HOLD_SAFE_SIDE] = 1

    return option_mask


def _option11_mask_to_composite26(mask: List[int]) -> List[int]:
    composite_mask = [0] * COMPOSITE_ACTION_SPACE_SHAPE[0]
    for option, enabled in enumerate(mask):
        if enabled:
            composite_mask[OPTION11_TO_COMPOSITE26.get(option, 5)] = 1

    if not any(composite_mask):
        composite_mask[OPTION11_TO_COMPOSITE26[OPTION_HOLD_SAFE_SIDE]] = 1

    return composite_mask


def _raw_mask_to_option(mask: List[int], obs: List[float]) -> List[int]:
    option_mask = []
    for option in range(MACRO_ACTION_SPACE_SHAPE[0]):
        raw_action = _option_to_raw_action(option, obs)
        option_mask.append(1 if _raw_action_allowed(raw_action, mask) else 0)

    if not any(option_mask):
        option_mask[OPTION_HOLD_SAFE_SIDE] = 1

    return option_mask


def _raw_mask_to_composite26(mask: List[int], obs: List[float]) -> List[int]:
    composite_mask = []
    for action in range(COMPOSITE_ACTION_SPACE_SHAPE[0]):
        raw_action = _composite26_to_raw_action(action, obs)
        composite_mask.append(1 if _raw_action_allowed(raw_action, mask) else 0)

    if not any(composite_mask):
        composite_mask[OPTION11_TO_COMPOSITE26[OPTION_HOLD_SAFE_SIDE]] = 1

    return composite_mask


def _raw_mask_to_lowlevel(mask: List[int], obs: List[float]) -> List[int]:
    lowlevel_mask = []
    for action in range(LOW_LEVEL_ACTION_SPACE_SHAPE[0]):
        raw_action = _lowlevel_to_raw_action(action, obs)
        lowlevel_mask.append(1 if _raw_action_allowed(raw_action, mask) else 0)

    if not any(lowlevel_mask):
        lowlevel_mask[0] = 1

    return lowlevel_mask


def _composite26_mask_to_lowlevel(mask: List[int], obs: List[float]) -> List[int]:
    lowlevel_mask = [0] * LOW_LEVEL_ACTION_SPACE_SHAPE[0]
    for action in range(COMPOSITE_ACTION_SPACE_SHAPE[0]):
        if action >= len(mask) or not mask[action]:
            continue
        lowlevel = _raw_action_to_lowlevel(_composite26_to_raw_action(action, obs), obs)
        lowlevel_mask[lowlevel] = 1

    if not any(lowlevel_mask):
        lowlevel_mask[0] = 1

    return lowlevel_mask


def _option11_mask_to_lowlevel(mask: List[int], obs: List[float]) -> List[int]:
    lowlevel_mask = [0] * LOW_LEVEL_ACTION_SPACE_SHAPE[0]
    for option in range(MACRO_ACTION_SPACE_SHAPE[0]):
        if option >= len(mask) or not mask[option]:
            continue
        lowlevel = _raw_action_to_lowlevel(_option_to_raw_action(option, obs), obs)
        lowlevel_mask[lowlevel] = 1

    if not any(lowlevel_mask):
        lowlevel_mask[0] = 1

    return lowlevel_mask


def _enable_raw_action(raw_mask: List[int], raw_action: List[int]) -> None:
    for branch_index, value in enumerate(raw_action):
        if branch_index >= len(RAW_ACTION_SPACE_SHAPE):
            continue
        value = int(value)
        if 0 <= value < RAW_ACTION_SPACE_SHAPE[branch_index]:
            raw_mask[RAW_BRANCH_OFFSETS[branch_index] + value] = 1


def _lowlevel_mask_to_raw(mask: List[int], obs: List[float]) -> List[int]:
    raw_mask = [0] * sum(RAW_ACTION_SPACE_SHAPE)
    for action in range(LOW_LEVEL_ACTION_SPACE_SHAPE[0]):
        if action < len(mask) and mask[action]:
            _enable_raw_action(raw_mask, _lowlevel_to_raw_action(action, obs))
    if not any(raw_mask):
        _enable_raw_action(raw_mask, [0] * len(RAW_ACTION_SPACE_SHAPE))
    return raw_mask


def _composite26_mask_to_raw(mask: List[int], obs: List[float]) -> List[int]:
    raw_mask = [0] * sum(RAW_ACTION_SPACE_SHAPE)
    for action in range(COMPOSITE_ACTION_SPACE_SHAPE[0]):
        if action < len(mask) and mask[action]:
            _enable_raw_action(raw_mask, _composite26_to_raw_action(action, obs))
    if not any(raw_mask):
        _enable_raw_action(raw_mask, _option_to_raw_action(OPTION_HOLD_SAFE_SIDE, obs))
    return raw_mask


def _option11_mask_to_raw(mask: List[int], obs: List[float]) -> List[int]:
    raw_mask = [0] * sum(RAW_ACTION_SPACE_SHAPE)
    for option in range(MACRO_ACTION_SPACE_SHAPE[0]):
        if option < len(mask) and mask[option]:
            _enable_raw_action(raw_mask, _option_to_raw_action(option, obs))
    if not any(raw_mask):
        _enable_raw_action(raw_mask, _option_to_raw_action(OPTION_HOLD_SAFE_SIDE, obs))
    return raw_mask


def _legacy_macro_mask_to_option(mask: List[int]) -> List[int]:
    option_mask = [0] * MACRO_ACTION_SPACE_SHAPE[0]
    for macro, enabled in enumerate(mask):
        if enabled:
            option_mask[_legacy_macro_to_option(macro, [])] = 1

    if not any(option_mask):
        option_mask[OPTION_HOLD_SAFE_SIDE] = 1

    return option_mask


def _composite26_to_raw_action(action: int, obs: List[float]) -> List[int]:
    if action == 24:
        return _option_to_raw_action(OPTION_USE_CLOSE_SKILL, obs)
    if action == 25:
        return _option_to_raw_action(OPTION_BIND_HEAL, obs)

    movement = int(action) % COMPOSITE_MOVEMENT_COUNT
    offense = int(action) // COMPOSITE_MOVEMENT_COUNT
    raw = _option_to_raw_action(movement, obs)

    if offense == 1:
        raw[3] = 1
        if raw[0] == MOVE_NONE:
            raw[0] = _horizontal_slash_direction(obs)
    elif offense == 2:
        raw[1] = LOOK_UP
        raw[3] = 1
    elif offense == 3:
        raw[7] = TOOL_DOWN if _abs_x(obs) > 0.11 else TOOL_UP
        if raw[0] == MOVE_NONE and not _facing_boss(obs):
            raw[0] = _direction_to_boss(obs)

    return raw


def _option_to_raw_action(option: int, obs: List[float]) -> List[int]:
    raw = [0] * len(RAW_ACTION_SPACE_SHAPE)
    toward = _direction_to_boss(obs)
    away = _opposite_direction(toward)

    if option == OPTION_KEEP_SWEET_SPOT:
        raw[0] = _keep_range_direction(obs)
    elif option == OPTION_RETREAT_FROM_BOSS_DODGE:
        raw[0] = away
    elif option == OPTION_DODGE_OVER_BOSS_HIGH_JUMP:
        raw[0] = toward
        raw[2] = 1
    elif option == OPTION_CROSS_UNDER_AIR_BOSS:
        raw[0] = toward
    elif option == OPTION_MOVE_TO_CENTER_SAFELY:
        raw[0] = _direction_to_center(obs)
    elif option == OPTION_HOLD_SAFE_SIDE:
        raw[0] = MOVE_NONE
    elif option == OPTION_PUNISH_WITH_SLASH:
        raw[0] = _horizontal_slash_direction(obs)
        raw[3] = 1
    elif option == OPTION_ANTI_AIR_SLASH:
        raw[1] = LOOK_UP
        raw[3] = 1
    elif option == OPTION_USE_CLOSE_SKILL:
        raw[7] = TOOL_NEUTRAL
        if not _facing_boss(obs):
            raw[0] = toward
    elif option == OPTION_USE_RANGED_TOOL:
        raw[7] = TOOL_DOWN if _abs_x(obs) > 0.11 else TOOL_UP
        if not _facing_boss(obs):
            raw[0] = toward
    elif option == OPTION_BIND_HEAL:
        raw[6] = 1

    return raw


def _raw_action_allowed(action: List[int], mask: List[int]) -> bool:
    if mask is None or len(mask) != sum(RAW_ACTION_SPACE_SHAPE):
        return True

    for branch_index, value in enumerate(action):
        offset = RAW_BRANCH_OFFSETS[branch_index]
        if value < 0 or value >= RAW_ACTION_SPACE_SHAPE[branch_index]:
            return False
        if mask[offset + value] == 0:
            return False

    return True


def _direction_to_boss(obs: List[float]) -> int:
    if not isinstance(obs, list) or len(obs) <= 5:
        return MOVE_NONE

    return MOVE_RIGHT if obs[5] >= obs[0] else MOVE_LEFT


def _abs_x(obs: List[float]) -> float:
    if not isinstance(obs, list) or len(obs) <= 5:
        return 0.0

    return abs(obs[5] - obs[0])


def _approach_direction(obs: List[float]) -> int:
    if not isinstance(obs, list) or len(obs) <= 5:
        return MOVE_NONE

    dx = obs[5] - obs[0]
    if abs(dx) <= 0.022:
        return _opposite_direction(_direction_to_boss(obs))

    return MOVE_RIGHT if dx >= 0.0 else MOVE_LEFT


def _keep_range_direction(obs: List[float]) -> int:
    if not isinstance(obs, list) or len(obs) <= 5:
        return MOVE_NONE

    dx = obs[5] - obs[0]
    abs_dx = abs(dx)
    if abs_dx > 0.076:
        return MOVE_RIGHT if dx >= 0.0 else MOVE_LEFT
    if abs_dx < 0.035:
        return MOVE_LEFT if dx >= 0.0 else MOVE_RIGHT
    if not _facing_boss(obs):
        return MOVE_RIGHT if dx >= 0.0 else MOVE_LEFT

    return MOVE_NONE


def _horizontal_slash_direction(obs: List[float]) -> int:
    if not isinstance(obs, list) or len(obs) <= 5:
        return MOVE_NONE

    dx = obs[5] - obs[0]
    abs_dx = abs(dx)
    toward = _direction_to_boss(obs)
    if abs_dx < 0.018:
        return _opposite_direction(toward)
    if not _facing_boss(obs) or abs_dx > 0.039:
        return toward
    return MOVE_NONE


def _jump_direction(obs: List[float]) -> int:
    center_direction = _direction_to_center(obs)
    if center_direction != MOVE_NONE and _far_from_center(obs):
        return center_direction

    if not isinstance(obs, list) or len(obs) <= 5:
        return MOVE_NONE

    if abs(obs[5] - obs[0]) <= 0.022:
        return _opposite_direction(_direction_to_boss(obs))

    return MOVE_NONE


def _within_keep_range(obs: List[float]) -> bool:
    if not isinstance(obs, list) or len(obs) <= 5:
        return False

    abs_dx = abs(obs[5] - obs[0])
    return 0.035 <= abs_dx <= 0.076


def _direction_to_center(obs: List[float]) -> int:
    if not isinstance(obs, list) or len(obs) == 0:
        return MOVE_NONE

    if obs[0] < 0.49:
        return MOVE_RIGHT
    if obs[0] > 0.51:
        return MOVE_LEFT
    return MOVE_NONE


def _far_from_center(obs: List[float]) -> bool:
    return isinstance(obs, list) and len(obs) > 0 and abs(obs[0] - 0.5) > 0.07


def _opposite_direction(direction: int) -> int:
    if direction == MOVE_LEFT:
        return MOVE_RIGHT
    if direction == MOVE_RIGHT:
        return MOVE_LEFT
    return MOVE_NONE


def _facing_boss(obs: List[float]) -> bool:
    if not isinstance(obs, list) or len(obs) <= 21:
        return True

    if _looks_like_current_lace_observation(obs) and len(obs) > 22:
        return obs[22] > 0.5

    boss_is_right = obs[5] >= obs[0]
    hero_facing_right = obs[21] > 0.5
    return hero_facing_right if boss_is_right else not hero_facing_right


def _looks_like_current_lace_observation(obs: List[float]) -> bool:
    size = len(obs)
    if size in (LACE_COMPRESSED_SINGLE_FRAME_SIZE, LACE_COMPRESSED_DENSE_OBS_SIZE):
        return True
    if size % LACE_DENSE_FRAME_STACK_SIZE == 0:
        return size // LACE_DENSE_FRAME_STACK_SIZE == LACE_COMPRESSED_SINGLE_FRAME_SIZE
    return False


def _estimate_boss_facing_right(obs: List[float]) -> bool:
    if not isinstance(obs, list) or len(obs) <= 7:
        return True

    boss_vel_x = obs[7]
    if boss_vel_x > 0.505:
        return True
    if boss_vel_x < 0.495:
        return False

    return obs[5] < obs[0]


def infer_boss_name(demo_dir: str, min_steps: int = 20) -> str:
    counts = {}
    for path in Path(demo_dir).rglob("*.jsonl"):
        records = _load_file_records(path)
        if len(records) < min_steps:
            continue
        boss_name = records[0].get("boss_name")
        if boss_name:
            counts[boss_name] = counts.get(boss_name, 0) + len(records)

    if not counts:
        raise ValueError(f"No valid demonstration files found in {demo_dir}")

    return max(counts.items(), key=lambda item: item[1])[0]


def load_demonstrations(
    demo_dir: str = DEFAULT_DEMO_ROOT,
    boss_name: Optional[str] = None,
    obs_size: Optional[int] = None,
    action_shape: Optional[List[int]] = None,
    min_steps: int = 20,
    max_samples: Optional[int] = None,
    drop_mask_violations: bool = False,
) -> DemonstrationDataset:
    root = Path(demo_dir)
    if not root.exists():
        raise FileNotFoundError(f"Demo directory not found: {root}")

    if boss_name is None:
        boss_name = infer_boss_name(str(root), min_steps=min_steps)

    target_boss = _normalize(boss_name)
    observations = []
    actions = []
    masks = []
    used_files = []

    for path in sorted(root.rglob("*.jsonl")):
        records = _load_file_records(path)
        if len(records) < min_steps:
            continue

        file_boss = records[0].get("boss_name", "")
        if _normalize(file_boss) != target_boss:
            continue

        adapted_observations = _adapt_observation_sequence(records, obs_size)
        file_used = False
        for record, obs in zip(records, adapted_observations):
            action = record.get("action")
            mask = record.get("action_mask")

            if not isinstance(obs, list) or not isinstance(action, list):
                continue
            if obs_size is not None and len(obs) != obs_size:
                continue

            action = _adapt_action(action, obs, action_shape, record)
            if action is None:
                raw_action = record.get("raw_action")
                if isinstance(raw_action, list):
                    action = _adapt_action(raw_action, obs, action_shape, record)
            if action is None:
                continue
            if action_shape is not None and any(value < 0 or value >= dim for value, dim in zip(action, action_shape)):
                continue
            mask = _adapt_mask(mask, obs, action_shape, record)
            if drop_mask_violations and mask is not None and action_shape is not None:
                if not _action_allowed(action, mask, action_shape):
                    continue

            observations.append(obs)
            actions.append(action)
            masks.append(mask)
            file_used = True

            if max_samples is not None and len(observations) >= max_samples:
                break

        if file_used:
            used_files.append(str(path))
        if max_samples is not None and len(observations) >= max_samples:
            break

    if not observations:
        raise ValueError(f"No demonstration samples matched boss '{boss_name}' in {root}")

    obs_arr = np.asarray(observations, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.int64)

    mask_arr = None
    if masks and all(mask is not None for mask in masks):
        mask_arr = np.asarray(masks, dtype=np.bool_)

    return DemonstrationDataset(
        observations=obs_arr,
        actions=action_arr,
        action_masks=mask_arr,
        files=used_files,
        boss_name=boss_name,
    )


def _load_file_records(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
