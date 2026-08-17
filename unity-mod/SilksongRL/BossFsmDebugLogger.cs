using System;
using System.Text;
using UnityEngine;

namespace SilksongRL
{
    public class BossFsmDebugLogger : MonoBehaviour
    {
        private const float LogIntervalSeconds = 0.5f;

        public bool Enabled { get; set; }

        private float nextLogAt;
        private HealthManager cachedBoss;
        private bool loggedInventory;

        public void ResetCache()
        {
            cachedBoss = null;
            loggedInventory = false;
            nextLogAt = 0f;
        }

        public void Tick(HealthManager boss)
        {
            if (!Enabled)
                return;

            if (Time.unscaledTime < nextLogAt)
                return;

            nextLogAt = Time.unscaledTime + LogIntervalSeconds;
            LogBossFsm(boss);
        }

        private void LogBossFsm(HealthManager boss)
        {
            if (boss == null)
            {
                RLManager.StaticLogger?.LogInfo("[BossFSM] Boss is null");
                cachedBoss = null;
                loggedInventory = false;
                return;
            }

            PlayMakerFSM[] fsms = boss.GetComponentsInChildren<PlayMakerFSM>(true);
            if (cachedBoss != boss)
            {
                cachedBoss = boss;
                loggedInventory = false;
            }

            if (!loggedInventory)
            {
                loggedInventory = true;
                LogInventory(boss, fsms);
            }

            StringBuilder active = new StringBuilder();
            active.Append($"[BossFSM] Active states for '{boss.name}': ");
            if (fsms == null || fsms.Length == 0)
            {
                active.Append("<none>");
            }
            else
            {
                bool wroteAny = false;
                foreach (PlayMakerFSM fsm in fsms)
                {
                    if (fsm == null)
                        continue;

                    if (wroteAny)
                        active.Append(" | ");

                    active.Append(SafeName(fsm.FsmName))
                        .Append("@")
                        .Append(GetPath(fsm.transform))
                        .Append("=")
                        .Append(SafeName(fsm.ActiveStateName));
                    wroteAny = true;
                }

                if (!wroteAny)
                    active.Append("<none>");
            }

            RLManager.StaticLogger?.LogInfo(active.ToString());
        }

        private void LogInventory(HealthManager boss, PlayMakerFSM[] fsms)
        {
            int count = fsms != null ? fsms.Length : 0;
            RLManager.StaticLogger?.LogInfo($"[BossFSM] Found {count} PlayMakerFSM components under '{boss.name}'");
            if (fsms == null)
                return;

            foreach (PlayMakerFSM fsm in fsms)
            {
                if (fsm == null)
                    continue;

                StringBuilder builder = new StringBuilder();
                builder.Append($"[BossFSM] FSM '{SafeName(fsm.FsmName)}' on '{GetPath(fsm.transform)}': active='{SafeName(fsm.ActiveStateName)}', states=");
                int stateCount = fsm.FsmStates != null ? fsm.FsmStates.Length : 0;
                builder.Append(stateCount);
                RLManager.StaticLogger?.LogInfo(builder.ToString());

                if (stateCount <= 0)
                    continue;

                StringBuilder states = new StringBuilder();
                states.Append($"[BossFSM] States for '{SafeName(fsm.FsmName)}' on '{GetPath(fsm.transform)}': ");
                for (int i = 0; i < fsm.FsmStates.Length; i++)
                {
                    if (i > 0)
                        states.Append(", ");
                    states.Append(SafeName(fsm.FsmStates[i]?.Name));
                }
                RLManager.StaticLogger?.LogInfo(states.ToString());
            }
        }

        private string GetPath(Transform transform)
        {
            if (transform == null)
                return "<null>";

            StringBuilder builder = new StringBuilder(transform.name);
            Transform parent = transform.parent;
            while (parent != null)
            {
                builder.Insert(0, parent.name + "/");
                parent = parent.parent;
            }

            return builder.ToString();
        }

        private string SafeName(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? "<none>" : value.Trim();
        }
    }
}
