# Training Workflow

This guide assumes the Python dependencies are installed and `SilksongRL.dll`
is present in the active BepInEx profile. The system controls a real game
process, so a repeatable save state is part of the environment setup.

## 1. Configure the Python Backend

Create `python-client/server_config.json` from the example:

```powershell
cd python-client
Copy-Item server_config.example.json server_config.json
python launch.py
```

The server waits on `localhost:8000` by default. Start it before enabling agent
control in the game.

## 2. Configure the BepInEx Plugin

Run the game once to generate `BepInEx/config/silksongrl.cfg`, then edit the
relevant values:

| Section/key | Purpose | Default |
| --- | --- | --- |
| `Connection.Host` | Python server host | `localhost` |
| `Connection.Port` | Python server port | `8000` |
| `Training.TargetBoss` | Encounter key such as `Lace_1` or `Lace_2` | `Lace_1` |
| `Training.StepInterval` | Observation/reward sampling interval | `0.05` |
| `Training.DecisionInterval` | Minimum interval between new policy decisions | `0.05` |
| `Training.StepMode` | Pause between decisions and advance fixed physics frames | `true` |
| `Training.StepModeFrames` | Physics frames advanced per sample in step mode | `3` |
| `Training.EvalMode` | Deterministic inference without training transitions | `false` |
| `Training.EvalStatsWindow` | Episodes per printed evaluation summary | `25` |
| `Recorder.DemoRoot` | Human demonstration output directory | Unity persistent data |
| `Debug.HitboxDebug` | Initial collider logger state | `false` |
| `Debug.BossFsmDebug` | Initial FSM logger state | `false` |

Restart the game after changing BepInEx configuration.

## 3. Prepare a Repeatable Encounter

1. Navigate to the target boss with agent control disabled.
2. Use Silksong.DebugMod to create a save state immediately before combat.
3. Put that save state in the DebugMod quick slot.
4. Bind quick-slot load to `F5`.
5. Verify that pressing `F5` restores the same hero position, resources, and
   boss state.

The episode manager simulates `F5` after terminal states. A bad or transitional
save state will produce noisy resets and unreliable training data.

## 4. Start Training

1. Start `python launch.py` and wait for `Waiting for connection...`.
2. Start Silksong through the profile containing BepInEx, DebugMod, and
   `SilksongRL.dll`.
3. Load the prepared save and enter the configured encounter.
4. Confirm the BepInEx console reports the expected boss, observation size,
   action space, and socket connection.
5. Press `P` to enable agent control.

Press `P` again to return control to the player. Do not manually control the
hero while an online training episode is active.

## 5. Runtime Controls

| Key | Function |
| --- | --- |
| `P` | Toggle agent control/training |
| `F5` | DebugMod quick-load reset; also used automatically between episodes |
| `F6` | Toggle human demonstration recording while agent control is off |
| `F7` | Toggle relevant Collider2D/hitbox logging every 0.5 seconds |
| `F8` | Toggle active boss PlayMaker FSM-state logging every 0.5 seconds |

The debug loggers are intended for encounter development and should normally
remain disabled during long training runs.

## 6. Record Human Demonstrations

1. Disable agent control with `P`.
2. Press `F6` and play the encounter manually.
3. Reset or finish episodes normally; the recorder writes one `.jsonl` file per
   episode.
4. Press `F6` again to stop recording and close the active file.

Copy the resulting boss folder into `python-client/demos/` before running BC or
predictor training. Demonstrations are local data and are ignored by Git.

## 7. Behavior-Cloning Warm Start

```powershell
cd python-client
python train_bc.py --demo-dir demos\lace_1 --boss-name "Lace Boss1" `
  --epochs 40 --respect-masks
```

Use `python train_bc.py --help` to inspect weighting options. BC initializes the
policy and Frame Encoder; online PPO remains responsible for optimizing the
actual reward in the game.

## 8. Evaluation

Set the following in `silksongrl.cfg` and restart the game:

```ini
[Training]
EvalMode = true
EvalStatsWindow = 25
```

Evaluation mode loads a checkpoint, uses deterministic action selection, and
does not store transitions or update model weights. The BepInEx console prints
per-episode outcomes plus win rate and mean boss HP remaining for each window.

## 9. Checkpoints and Local Data

- PPO: `python-client/models/<normalized-boss-name>/`
- DQN: `python-client/dqn_models/<model-name>/`
- Future predictor: `python-client/predictors/<normalized-boss-name>/`
- Demonstrations: `python-client/demos/<boss-key>/`
- Optional agent traces: `python-client/agent_traces/<boss-key>/`

These paths are excluded from source control. Back them up separately if an
experiment must be preserved.

## Troubleshooting

- **The server waits forever:** the game plugin has not connected, host/port do
  not match, or a firewall is blocking localhost.
- **Observation/action dimensions mismatch:** the checkpoint was trained with a
  different encounter, observation schema, frame stack, or action space.
- **Episodes reset inconsistently:** recreate the save state outside scene
  transitions and increase `F5RetryInterval` if DebugMod is still loading.
- **The agent presses wrong controls:** this system emits game inputs and assumes
  compatible bindings.
- **Training timing changes:** keep `StepInterval`, `DecisionInterval`, step
  mode, and timescale consistent with the checkpoint being evaluated.
