using System;
using System.Globalization;
using System.IO;
using System.Text;
using InControl;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace SilksongRL
{
    public class DemonstrationRecorder : MonoBehaviour
    {
        private const float InputThreshold = 0.3f;

        private string demoRoot;
        private float stepInterval;
        private bool isEnabled;
        private StreamWriter writer;
        private string activeEpisodeId;
        private string activeBossKey;
        private int stepIndex;
        private float lastSampleTime;
        private bool waitingForNewEpisode;

        private bool pendingJump;
        private bool pendingAttack;
        private bool pendingDash;
        private bool pendingNeedle;
        private bool pendingBind;
        private ToolAction pendingTool = ToolAction.None;

        public void Initialize(string rootPath, float interval)
        {
            demoRoot = string.IsNullOrWhiteSpace(rootPath)
                ? Path.Combine(Application.persistentDataPath, "SilksongRL_Demos")
                : rootPath;
            stepInterval = Mathf.Max(0.01f, interval);
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.F6))
            {
                ToggleRecording();
            }

            if (!isEnabled || RLManager.isAgentControlEnabled || Time.timeScale <= Mathf.Epsilon)
                return;

            AccumulateSemanticInput();
        }

        private void FixedUpdate()
        {
            if (!isEnabled)
                return;

            if (RLManager.isAgentControlEnabled)
            {
                CloseEpisode("agent_control_enabled");
                return;
            }

            if (Time.fixedTime - lastSampleTime < stepInterval)
                return;

            lastSampleTime = Time.fixedTime;
            RecordStepIfReady();
        }

        private void OnDestroy()
        {
            CloseEpisode("recorder_destroyed");
        }

        private void ToggleRecording()
        {
            isEnabled = !isEnabled;
            ClearPendingInputs();

            if (isEnabled)
            {
                lastSampleTime = Time.fixedTime;
                waitingForNewEpisode = false;
                RLManager.CurrentEncounter?.ResetObservationHistory();
                RLManager.StaticLogger?.LogInfo($"[DemoRecorder] Recording enabled. Output root: {demoRoot}");
            }
            else
            {
                CloseEpisode("manual_stop");
                RLManager.StaticLogger?.LogInfo("[DemoRecorder] Recording disabled");
            }
        }

        private void AccumulateSemanticInput()
        {
            HeroActions actions = GetHeroActions();
            if (actions == null)
                return;

            pendingJump |= actions.Jump.IsPressed || actions.Jump.WasPressed;
            pendingAttack |= actions.Attack.IsPressed || actions.Attack.WasPressed;
            pendingDash |= actions.Dash.IsPressed || actions.Dash.WasPressed;
            pendingNeedle |= actions.SuperDash.IsPressed || actions.SuperDash.WasPressed;
            pendingBind |= actions.Cast.IsPressed || actions.Cast.WasPressed;

            if (actions.QuickCast.IsPressed || actions.QuickCast.WasPressed)
            {
                pendingTool = ReadToolAction(actions);
            }
        }

        private void RecordStepIfReady()
        {
            IBossEncounter encounter = RLManager.CurrentEncounter;
            HeroController hero = RLManager.Hero;
            HealthManager boss = RLManager.Boss;

            if (encounter == null || hero == null || boss == null || !encounter.IsEncounterMatch(boss))
            {
                CloseEpisode("missing_hero_or_boss");
                ClearPendingInputs();
                return;
            }

            if (Time.timeScale <= Mathf.Epsilon)
            {
                ClearPendingInputs();
                return;
            }

            float[] observation = encounter.ExtractObservationArray(hero, boss);
            if (observation == null)
            {
                ClearPendingInputs();
                return;
            }

            Action action = ReadSemanticAction();
            action.macro = ActionManager.InferMacroAction(action, hero, boss, encounter);
            int[] actionMask = encounter.GetActionMask(hero, boss);
            bool done = IsTerminal(hero, boss, out string terminalReason);

            if (done && writer == null)
            {
                waitingForNewEpisode = true;
                ClearPendingInputs();
                return;
            }

            if (waitingForNewEpisode)
            {
                waitingForNewEpisode = false;
                ClearPendingInputs();
            }

            EnsureEpisodeOpen(encounter, boss);
            WriteRecord(encounter, hero, boss, observation, action, actionMask, done, terminalReason);
            stepIndex++;

            ClearPendingInputs();

            if (done)
            {
                waitingForNewEpisode = true;
                CloseEpisode(terminalReason);
            }
        }

        private Action ReadSemanticAction()
        {
            HeroActions actions = GetHeroActions();
            Action action = new Action();

            if (actions != null)
            {
                Vector2 move = actions.MoveVector.Value;
                if (move.x < -InputThreshold || actions.Left.IsPressed)
                    action.move = MoveDirection.Left;
                else if (move.x > InputThreshold || actions.Right.IsPressed)
                    action.move = MoveDirection.Right;

                if (move.y > InputThreshold || actions.Up.IsPressed)
                    action.look = LookDirection.Up;
                else if (move.y < -InputThreshold || actions.Down.IsPressed)
                    action.look = LookDirection.Down;
            }

            action.jump = pendingJump;
            action.attack = pendingAttack;
            action.dash = pendingDash;
            action.needle = pendingNeedle;
            action.bind = pendingBind;
            action.tool = pendingTool;
            return action;
        }

        private ToolAction ReadToolAction(HeroActions actions)
        {
            Vector2 move = actions.MoveVector.Value;
            if (move.y > InputThreshold || actions.Up.IsPressed)
                return ToolAction.Up;
            if (move.y < -InputThreshold || actions.Down.IsPressed)
                return ToolAction.Down;

            return ToolAction.Neutral;
        }

        private HeroActions GetHeroActions()
        {
            try
            {
                InputHandler handler = ManagerSingleton<InputHandler>.Instance;
                return handler != null ? handler.inputActions : null;
            }
            catch
            {
                return null;
            }
        }

        private bool IsTerminal(HeroController hero, HealthManager boss, out string reason)
        {
            if (boss == null || boss.hp <= 0)
            {
                reason = "boss_dead";
                return true;
            }

            if (hero != null && hero.playerData != null && (hero.playerData.health <= 0 || hero.cState.dead))
            {
                reason = "hero_dead";
                return true;
            }

            reason = "";
            return false;
        }

        private void EnsureEpisodeOpen(IBossEncounter encounter, HealthManager boss)
        {
            string bossKey = GetBossKey(encounter, boss);
            if (writer != null && activeBossKey == bossKey)
                return;

            CloseEpisode("boss_changed");

            activeBossKey = bossKey;
            activeEpisodeId = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            stepIndex = 0;

            string folder = Path.Combine(demoRoot, bossKey);
            Directory.CreateDirectory(folder);
            string path = Path.Combine(folder, activeEpisodeId + ".jsonl");

            writer = new StreamWriter(path, append: false, Encoding.UTF8);
            writer.AutoFlush = true;
            RLManager.StaticLogger?.LogInfo($"[DemoRecorder] Started episode recording: {path}");
        }

        private string GetBossKey(IBossEncounter encounter, HealthManager boss)
        {
            string key = RLManager.CurrentTargetBossKey;
            if (string.IsNullOrWhiteSpace(key))
                key = encounter != null ? encounter.GetEncounterName() : boss != null ? boss.name : "unknown_boss";

            return NormalizeName(key);
        }

        private string NormalizeName(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return "unknown";

            StringBuilder builder = new StringBuilder(value.Length);
            bool lastWasSeparator = false;
            foreach (char c in value.Trim().ToLowerInvariant())
            {
                if (char.IsLetterOrDigit(c))
                {
                    builder.Append(c);
                    lastWasSeparator = false;
                }
                else if (!lastWasSeparator)
                {
                    builder.Append('_');
                    lastWasSeparator = true;
                }
            }

            return builder.ToString().Trim('_');
        }

        private void WriteRecord(
            IBossEncounter encounter,
            HeroController hero,
            HealthManager boss,
            float[] observation,
            Action action,
            int[] actionMask,
            bool done,
            string terminalReason)
        {
            StringBuilder json = new StringBuilder(4096);
            json.Append('{');
            AppendJsonField(json, "version", 7).Append(',');
            AppendJsonField(json, "source", "human_semantic_macro_incontrol").Append(',');
            AppendJsonArrayField(json, "action_space_shape", ActionManager.GetActionSpaceShape()).Append(',');
            AppendJsonField(json, "boss_key", activeBossKey).Append(',');
            AppendJsonField(json, "boss_name", encounter.GetEncounterName()).Append(',');
            AppendJsonField(json, "scene", SceneManager.GetActiveScene().name).Append(',');
            AppendJsonField(json, "episode_id", activeEpisodeId).Append(',');
            AppendJsonField(json, "step", stepIndex).Append(',');
            AppendJsonField(json, "time", Time.time).Append(',');
            AppendJsonField(json, "done", done).Append(',');
            AppendJsonField(json, "terminal_reason", terminalReason ?? "").Append(',');
            AppendJsonField(json, "observation_size", observation.Length).Append(',');
            AppendJsonArrayField(json, "observation", observation).Append(',');
            AppendJsonArrayField(json, "action", ActionManager.ActionToArray(action)).Append(',');
            AppendJsonArrayField(json, "raw_action", ActionManager.ActionToRawArray(action)).Append(',');
            AppendJsonArrayField(json, "action_mask", actionMask).Append(',');
            AppendJsonField(json, "hero_hp", hero.playerData != null ? hero.playerData.health : 0).Append(',');
            AppendJsonField(json, "boss_hp", boss != null ? boss.hp : 0).Append(',');
            AppendJsonField(json, "silk", PlayerData.HasInstance ? PlayerData.instance.silk : 0);
            json.Append('}');

            writer.WriteLine(json.ToString());
        }

        private StringBuilder AppendJsonField(StringBuilder json, string name, string value)
        {
            AppendJsonName(json, name);
            json.Append('"').Append(EscapeJson(value)).Append('"');
            return json;
        }

        private StringBuilder AppendJsonField(StringBuilder json, string name, int value)
        {
            AppendJsonName(json, name);
            json.Append(value.ToString(CultureInfo.InvariantCulture));
            return json;
        }

        private StringBuilder AppendJsonField(StringBuilder json, string name, float value)
        {
            AppendJsonName(json, name);
            json.Append(value.ToString("R", CultureInfo.InvariantCulture));
            return json;
        }

        private StringBuilder AppendJsonField(StringBuilder json, string name, bool value)
        {
            AppendJsonName(json, name);
            json.Append(value ? "true" : "false");
            return json;
        }

        private StringBuilder AppendJsonArrayField(StringBuilder json, string name, float[] values)
        {
            AppendJsonName(json, name);
            json.Append('[');
            for (int i = 0; i < values.Length; i++)
            {
                if (i > 0)
                    json.Append(',');
                json.Append(values[i].ToString("R", CultureInfo.InvariantCulture));
            }
            json.Append(']');
            return json;
        }

        private StringBuilder AppendJsonArrayField(StringBuilder json, string name, int[] values)
        {
            AppendJsonName(json, name);
            json.Append('[');
            for (int i = 0; i < values.Length; i++)
            {
                if (i > 0)
                    json.Append(',');
                json.Append(values[i].ToString(CultureInfo.InvariantCulture));
            }
            json.Append(']');
            return json;
        }

        private void AppendJsonName(StringBuilder json, string name)
        {
            json.Append('"').Append(name).Append("\":");
        }

        private string EscapeJson(string value)
        {
            return (value ?? "")
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n");
        }

        private void ClearPendingInputs()
        {
            pendingJump = false;
            pendingAttack = false;
            pendingDash = false;
            pendingNeedle = false;
            pendingBind = false;
            pendingTool = ToolAction.None;
        }

        private void CloseEpisode(string reason)
        {
            if (writer == null)
                return;

            writer.Flush();
            writer.Dispose();
            writer = null;
            RLManager.CurrentEncounter?.ResetObservationHistory();
            RLManager.StaticLogger?.LogInfo($"[DemoRecorder] Closed episode {activeEpisodeId} ({reason}), steps: {stepIndex}");
            activeEpisodeId = null;
            activeBossKey = null;
            stepIndex = 0;
        }
    }
}
