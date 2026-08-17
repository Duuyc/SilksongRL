from __future__ import annotations

from typing import Iterable, List, Optional

import torch


DEFAULT_BRANCH_WEIGHTS = [0.5, 0.4, 2.0, 8.0, 2.0, 1.0, 2.0, 2.0]
DEFAULT_NONZERO_BOOSTS = [0.0, 0.0, 10.0, 4.0, 6.0, 2.0, 40.0, 8.0]
DEFAULT_MACRO_ACTION_VALUE_WEIGHTS = [
    1.5,  # KeepSweetSpot
    6,  # RetreatFromBossDodge
    5,  # DodgeOverBossHighJump
    3,  # CrossUnderAirBoss
    1.5,  # MoveToCenterSafely
    2,  # HoldSafeSide
    15,  # PunishWithSlash
    8,  # AntiAirSlash
    6.0,  # UseCloseSkill
    12,  # UseRangedTool
    8,  # BindHeal
]
DEFAULT_COMPOSITE26_ACTION_VALUE_WEIGHTS = [
    1.0,  # KeepSweetSpot
    3.5,  # RetreatFromBossDodge
    3.0,  # DodgeOverBossHighJump
    2.5,  # CrossUnderAirBoss
    1.0,  # MoveToCenterSafely
    1.2,  # HoldSafeSide
    14.0,  # KeepSweetSpot+Slash
    18.0,  # Retreat+Slash
    18.0,  # DodgeOverBoss+Slash
    14.0,  # CrossUnder+Slash
    12.0,  # MoveToCenter+Slash
    18.0,  # Hold+Slash
    10.0,  # KeepSweetSpot+AntiAir
    12.0,  # Retreat+AntiAir
    12.0,  # DodgeOverBoss+AntiAir
    14.0,  # CrossUnder+AntiAir
    10.0,  # MoveToCenter+AntiAir
    14.0,  # Hold+AntiAir
    8.0,  # KeepSweetSpot+Tool
    10.0,  # Retreat+Tool
    10.0,  # DodgeOverBoss+Tool
    10.0,  # CrossUnder+Tool
    8.0,  # MoveToCenter+Tool
    12.0,  # Hold+Tool
    4.0,  # UseCloseSkill
    6.0,  # BindHeal
]
DEFAULT_LOW_LEVEL30_ACTION_VALUE_WEIGHTS = [
    0.5,   # Idle
    1.0,   # MoveLeft
    1.0,   # MoveRight
    0.5,   # LookUp
    0.5,   # LookDown
    2.0,   # Jump
    2.5,   # JumpLeft
    2.5,   # JumpRight
    2.0,   # Dash
    3.0,   # DashLeft
    3.0,   # DashRight
    8.0,   # Attack
    10.0,  # MoveLeftAttack
    10.0,  # MoveRightAttack
    8.0,   # UpAttack
    6.0,   # DownAttack
    9.0,   # JumpAttack
    10.0,  # JumpLeftAttack
    10.0,  # JumpRightAttack
    7.0,   # DashLeftAttack
    7.0,   # DashRightAttack
    2.0,   # Needle
    6.0,   # BindHeal
    5.0,   # ToolNeutral
    8.0,   # ToolUp
    8.0,   # ToolDown
    4.0,   # MoveLeftTool
    4.0,   # MoveRightTool
    4.0,   # JumpLeftTool
    4.0,   # JumpRightTool
]


def parse_branch_weights(value: Optional[str], expected_size: int) -> List[float]:
    return _parse_branch_values(value, expected_size, DEFAULT_BRANCH_WEIGHTS, "BC branch weights")


def parse_nonzero_boosts(value: Optional[str], expected_size: int) -> List[float]:
    return _parse_branch_values(value, expected_size, DEFAULT_NONZERO_BOOSTS, "BC nonzero boosts")


def parse_action_value_weights(value: Optional[str], action_shape: List[int]) -> Optional[List[List[float]]]:
    """Parse per-action weights, currently used for the single macro-action branch."""
    if len(action_shape) == 1 and action_shape[0] == 11 and (value is None or not value.strip()):
        return [DEFAULT_MACRO_ACTION_VALUE_WEIGHTS[:action_shape[0]]]
    if len(action_shape) == 1 and action_shape[0] == 26 and (value is None or not value.strip()):
        return [DEFAULT_COMPOSITE26_ACTION_VALUE_WEIGHTS[:action_shape[0]]]
    if len(action_shape) == 1 and action_shape[0] == 30 and (value is None or not value.strip()):
        return [DEFAULT_LOW_LEVEL30_ACTION_VALUE_WEIGHTS[:action_shape[0]]]

    if value is None or not value.strip():
        return None

    branches: List[List[float]] = []
    branch_specs = [part.strip() for part in value.split(";") if part.strip()]
    if len(branch_specs) == 1 and len(action_shape) == 1:
        branch_specs = branch_specs * len(action_shape)

    if len(branch_specs) != len(action_shape):
        raise ValueError(f"Expected {len(action_shape)} BC action-value weight branches, got {len(branch_specs)}")

    for spec, dim in zip(branch_specs, action_shape):
        weights = [1.0] * dim
        if ":" in spec:
            for item in spec.split(","):
                if not item.strip():
                    continue
                index_text, weight_text = item.split(":", 1)
                index = int(index_text.strip())
                if index < 0 or index >= dim:
                    raise ValueError(f"Action index {index} out of range for branch size {dim}")
                weights[index] = float(weight_text.strip())
        else:
            values = [float(part.strip()) for part in spec.split(",") if part.strip()]
            if len(values) != dim:
                raise ValueError(f"Expected {dim} action-value weights, got {len(values)}")
            weights = values

        branches.append(weights)

    return branches


def _parse_branch_values(
    value: Optional[str],
    expected_size: int,
    default_values: List[float],
    label: str,
) -> List[float]:
    if value is None or not value.strip():
        values = default_values[:expected_size]
    else:
        parts = [float(part.strip()) for part in value.split(",") if part.strip()]
        if len(parts) == 1:
            values = parts * expected_size
        else:
            values = parts

    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} {label}, got {len(values)}")

    return values


def weighted_multidiscrete_bc_loss(
    distribution,
    actions: torch.Tensor,
    branch_weights: Optional[Iterable[float]] = None,
    nonzero_boosts: Optional[Iterable[float] | float] = None,
    action_value_weights: Optional[Iterable[Optional[Iterable[float]]]] = None,
) -> torch.Tensor:
    """Weighted negative log likelihood for SB3-contrib MultiDiscrete policies."""
    if not hasattr(distribution, "distributions"):
        return -distribution.log_prob(actions).mean()

    branches = distribution.distributions
    if branch_weights is None:
        branch_weights = [1.0] * len(branches)
    else:
        branch_weights = list(branch_weights)

    if len(branch_weights) != len(branches):
        raise ValueError(f"Expected {len(branches)} branch weights, got {len(branch_weights)}")

    if nonzero_boosts is None:
        nonzero_boosts = DEFAULT_NONZERO_BOOSTS[:len(branches)]
    elif isinstance(nonzero_boosts, (int, float)):
        nonzero_boosts = [float(nonzero_boosts)] * len(branches)
    else:
        nonzero_boosts = list(nonzero_boosts)

    if len(nonzero_boosts) != len(branches):
        raise ValueError(f"Expected {len(branches)} nonzero boosts, got {len(nonzero_boosts)}")

    if action_value_weights is not None:
        action_value_weights = list(action_value_weights)
        if len(action_value_weights) != len(branches):
            raise ValueError(f"Expected {len(branches)} action-value weight branches, got {len(action_value_weights)}")

    weighted_loss = None
    normalizer = None
    for branch_index, branch_distribution in enumerate(branches):
        branch_action = actions[:, branch_index]
        log_prob = branch_distribution.log_prob(branch_action)

        weight = torch.full_like(log_prob, float(branch_weights[branch_index]))
        nonzero_boost = float(nonzero_boosts[branch_index])
        if nonzero_boost > 0:
            weight = weight * torch.where(
                branch_action != 0,
                torch.full_like(weight, 1.0 + nonzero_boost),
                torch.ones_like(weight),
            )

        if action_value_weights is not None and action_value_weights[branch_index] is not None:
            value_weights = torch.as_tensor(
                list(action_value_weights[branch_index]),
                dtype=weight.dtype,
                device=weight.device,
            )
            action_count = int(branch_distribution.logits.shape[-1])
            if value_weights.numel() != action_count:
                raise ValueError(
                    f"Expected {action_count} action-value weights, got {value_weights.numel()}"
                )
            weight = weight * value_weights[branch_action]

        term = -(log_prob * weight).sum()
        weighted_loss = term if weighted_loss is None else weighted_loss + term
        normalizer = weight.sum() if normalizer is None else normalizer + weight.sum()

    return weighted_loss / normalizer.clamp_min(1.0)
