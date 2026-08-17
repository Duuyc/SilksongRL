# SilksongRL: Structured-Observation Boss Combat RL

An experimental reinforcement-learning system that trains an agent against
bosses in a real Hollow Knight: Silksong game process. A BepInEx mod extracts
structured game state, computes rewards and action masks, and exchanges steps
with a Python training backend over a socket connection.

This repository is a course-project and portfolio extension of
[jimmie-jams/SilksongRL](https://github.com/jimmie-jams/SilksongRL). It retains
the upstream Unity/BepInEx foundation and adds a substantially expanded
observation, action, training, recording, and evaluation stack. See
[Attribution](#attribution) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Results

The final reported Lace Boss1 experiment used the model family
`lace_boss1_move_offense26_fsmelapsed_frameenc_obs7424` with 26 macro actions,
a 16-frame structured observation stack, and a 512-dimensional Frame Encoder.

| Metric | Result |
| --- | ---: |
| Best 25-episode training-window boss HP remaining | 29.81% |
| Best 25-episode training-window win rate | 24% |
| Deterministic evaluation episodes | 100 |
| Deterministic evaluation win rate | 11% |
| Deterministic evaluation mean boss HP remaining | 34.02% |

These numbers describe one real-game experiment, not a claim of a fully solved
or universally reproducible boss encounter. Game timing, save-state setup,
hardware, mod versions, and random boss behavior all affect results.

**Gameplay demo:** [Silksong reinforcement-learning agent vs. Lace Boss1
(Bilibili)](https://www.bilibili.com/video/BV1hSJ36TEe7/)

## System Architecture

```text
Silksong / Unity
      |
      v
BepInEx + Harmony patches
  - structured observations
  - rewards and action masks
  - episode/reset logic
      |
      v
socket protocol  <---->  Python training backend
                         - Maskable PPO (default)
                         - Dueling Double DQN experiment
                         - Frame Encoder
                         - BC and future prediction tools
      |
      v
26 high-level options -> ActionManager -> key-level game input
```

## Main Extensions in This Fork

- **26-option macro action space:** six movement/defense semantics combined
  with no attack, horizontal slash, anti-air slash, or ranged tool, plus close
  skill and bind/heal actions.
- **Structured temporal observations:** kinematics, resources, semantic FSM
  flags, FSM elapsed time, 32 raycasts, and collider/hitbox features.
- **Frame Encoder:** a shared per-frame MLP compresses each frame before a
  temporal MLP produces a 512-dimensional policy feature.
- **Maskable PPO:** environment-side masks remove actions that are invalid or
  clearly unsafe in the current state.
- **Behavior cloning:** human `.jsonl` demonstrations can warm-start the policy
  before online RL.
- **Optional future predictor:** predicts short-horizon boss displacement,
  danger, punish windows, and semantic FSM flags.
- **Experimental DQN backend:** dueling Double DQN with prioritized replay,
  n-step returns, and elite replay support.
- **Training support:** running reward normalization, best-checkpoint selection,
  rollback, deterministic evaluation statistics, and agent trace recording.
- **Debug tooling:** `F6` demonstration recording, `F7` collider logging, and
  `F8` boss FSM logging.
- **Standard benchmark:** an ALE/Pong RAM training script provides a lightweight
  sanity check outside the game integration.

## Repository Layout

```text
python-client/                 Python socket server and RL training code
  rl_core.py                   Maskable PPO lifecycle and inference
  dqn_core.py                  Experimental DQN backend
  frame_encoder.py             Structured temporal feature extractor
  demo_dataset.py              Demonstration parsing and action mapping
  train_bc.py                  PPO behavior-cloning warm start
  future_predictor.py          Short-horizon prediction model/runtime
  train_future_predictor.py    Predictor training and validation
  train_pong_ram_benchmark.py  ALE/Pong RAM benchmark
unity-mod/SilksongRL/          BepInEx plugin and encounter definitions
option-action-list.md          Macro-action reference
HOW_TO_TRAIN.md                End-to-end runtime workflow
```

Generated demonstrations, checkpoints, predictor weights, traces, logs, game
files, and save states are intentionally excluded from Git.

## Prerequisites

- A legally obtained Hollow Knight: Silksong installation.
- [BepInEx 5.4.x for Silksong](https://thunderstore.io/c/hollow-knight-silksong/p/BepInEx/BepInExPack_Silksong/).
- [Silksong.DebugMod](https://github.com/hk-speedrunning/Silksong.DebugMod)
  for the save-state reset workflow.
- Python 3.11.
- For C# builds: .NET Framework 4.7.2 targeting pack and Visual Studio/MSBuild.

## Python Setup

```powershell
cd python-client
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item server_config.example.json server_config.json
python launch.py
```

`server_config.json` selects the socket implementation and training backend:

```json
{
  "transport": "socket_sync",
  "algorithm": "ppo",
  "host": "localhost",
  "port": 8000
}
```

Use `"algorithm": "dqn"` to run the experimental DQN backend. Maskable PPO is
the default and is the backend used for the final Lace Boss1 model.

## Build and Install the Unity Mod

1. Copy the local project settings template:

   ```powershell
   Copy-Item unity-mod\SilksongRL\SilksongRL.csproj.user.example `
     unity-mod\SilksongRL\SilksongRL.csproj.user
   ```

2. Edit `SilksongRL.csproj.user` and set `GameDir` to your game installation.
   You may instead set the `SILKSONG_GAME_DIR` environment variable.

3. Build `unity-mod/SilksongRL.sln` in Visual Studio or with MSBuild.

4. Copy the generated `SilksongRL.dll` into the active profile's
   `BepInEx/plugins/` directory.

The project references BepInEx and game-managed assemblies from your local game
directory. Those proprietary/runtime binaries are not included here.

## Run Training

The complete game, save-state, server, and hotkey workflow is documented in
[HOW_TO_TRAIN.md](HOW_TO_TRAIN.md). In short:

1. Prepare a boss save state and bind DebugMod quick-load to `F5`.
2. Start `python-client/launch.py`.
3. Start the game and enter the configured encounter.
4. Press `P` to enable agent control.

The BepInEx file `BepInEx/config/silksongrl.cfg` controls the encounter,
sampling interval, fixed-step mode, evaluation mode, and recorder output.

## Behavior Cloning and Predictor Training

Record human play with `F6` while agent control is disabled, then run:

```powershell
cd python-client
python train_bc.py --demo-dir demos\lace_1 --boss-name "Lace Boss1" `
  --epochs 40 --respect-masks
```

Inspect all available weighting and dataset options with:

```powershell
python train_bc.py --help
python train_future_predictor.py --help
python train_dqn_bc.py --help
```

The online PPO behavior-cloning regularizer and future predictor are disabled
by default. They can be enabled through the `SILKSONGRL_BC_COEF` and
`SILKSONGRL_FUTURE_PREDICTOR` environment variables after compatible local
data/model files have been prepared.

## ALE/Pong RAM Sanity Check

```powershell
cd python-client
python train_pong_ram_benchmark.py --help
```

This benchmark checks whether the PPO and temporal encoder stack can learn in a
standard RAM-observation task. It is a sanity check, not a substitute for
Silksong boss evaluation.

## Configuration Notes

- Final experiments used `StepInterval = 0.05` and
  `DecisionInterval = 0.05` seconds.
- `EvalMode = true` uses deterministic inference and does not store training
  transitions. Evaluation summaries are printed every `EvalStatsWindow`
  episodes.
- PPO checkpoints are written below `python-client/models/`; DQN checkpoints
  use `python-client/dqn_models/`. Both directories are ignored by Git.
- The Python defaults can be overridden with `SILKSONGRL_*` environment
  variables defined near the top of `rl_core.py` and `dqn_core.py`.

## Attribution

This is a fork of [jimmie-jams/SilksongRL](https://github.com/jimmie-jams/SilksongRL),
distributed under the MIT License. The original copyright notice is retained.
The structured-observation pipeline, macro-action system, temporal encoder,
behavior-cloning and prediction utilities, alternative DQN experiments, and
training/evaluation extensions in this fork were developed as a course project
and portfolio implementation.

This project is unofficial and is not affiliated with or endorsed by Team
Cherry. It does not distribute game assets or binaries.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
