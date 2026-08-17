import os


ALGORITHM = os.environ.get("SILKSONGRL_ALGORITHM", "ppo").strip().lower()

if ALGORITHM in ("dqn", "masked_dqn", "rainbow"):
    from dqn_core import (
        initialize_model,
        get_action,
        store_transition,
    )
elif ALGORITHM in ("ppo", "maskable_ppo"):
    from rl_core import (
        initialize_model,
        get_action,
        store_transition,
    )
else:
    raise ValueError(
        f"Unknown SILKSONGRL_ALGORITHM={ALGORITHM!r}; expected 'dqn' or 'ppo'"
    )


print(f"[TrainingBackend] Using algorithm: {ALGORITHM}")
