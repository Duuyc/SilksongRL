using UnityEngine;

namespace SilksongRL
{
    /// <summary>
    /// Manages the lifecycle of training episodes including death detection,
    /// reset sequences, and state transitions.
    /// </summary>
    public class TrainingEpisodeManager
    {
        // Episode state machine
        public enum EpisodeState
        {
            Training,        // Normal training mode
            HeroDead,        // Hero died, need to reset
            BossDead,        // Boss died, need to reset
            HeroStuck        // Hero stuck (e.g., below ground), need to force reset
        }

        public EpisodeState CurrentState { get; private set; }

        private IBossEncounter encounter;

        // TO DO 
        // MAKE STUCK STEP THRESHOLD CONFIGURABLE BY EACH ENCOUNTER
        private const int STUCK_STEP_THRESHOLD = 5000;
        private int consecutiveStuckSteps = 0;
        
        private bool hasTriggeredReset = false;
        private bool hasPressedF5 = false;
        private bool bossSeenThisEpisode = false;
        //假如我需要同时循环训练多个boss，则游戏内的切换boss逻辑需要依赖于不止一个F5
        //可能需要更多按键
        private float resetSequenceStartTime = 0f;
        private float f5PressTime = 0f;
        private int f5PressAttempts = 0;
        private const float DefaultResetDelayDuration = 1.0f;
        private const float HeroDeathResetDelayDuration = 4.5f;
        private const float ResetSettleDuration = 2.5f;
        private const float DefaultF5RetryInterval = 5f;
        private readonly float f5RetryInterval;

        public System.Action<KeyCode> OnSimulateKeyPress;
        public System.Action OnResetComplete;

        public TrainingEpisodeManager(IBossEncounter encounter, float f5RetryInterval = DefaultF5RetryInterval)
        {
            this.encounter = encounter;
            this.f5RetryInterval = Mathf.Max(2.5f, f5RetryInterval);
            CurrentState = EpisodeState.Training;
        }

        /// <summary>
        /// Updates the episode state based on current game conditions.
        /// Should be called every fixed update.
        /// </summary>
        public void UpdateEpisodeState(HeroController hero, HealthManager boss)
        {
            if (hero == null)
                return;

            if (boss != null)
            {
                bossSeenThisEpisode = true;
            }

            if (CurrentState == EpisodeState.Training)
            {
                if ((boss == null && bossSeenThisEpisode) || (boss != null && boss.hp <= 0))
                {
                    CurrentState = EpisodeState.BossDead;
                    RLManager.StaticLogger?.LogInfo("[TrainingEpisodeManager] Boss defeated detected");
                    consecutiveStuckSteps = 0;
                    return;
                }

                if (IsHeroDead(hero))
                {
                    CurrentState = EpisodeState.HeroDead;
                    RLManager.StaticLogger?.LogInfo("[TrainingEpisodeManager] Hero death detected");
                    consecutiveStuckSteps = 0;
                    return;
                }

                if (encounter.IsHeroStuck(hero))
                {
                    consecutiveStuckSteps++;
                    if (consecutiveStuckSteps >= STUCK_STEP_THRESHOLD)
                    {
                        CurrentState = EpisodeState.HeroStuck;
                        RLManager.StaticLogger?.LogInfo($"[TrainingEpisodeManager] Hero stuck for {consecutiveStuckSteps} steps - triggering reset");
                        consecutiveStuckSteps = 0;
                    }
                }
                else
                {
                    consecutiveStuckSteps = 0;
                }
            }
        }

        /// <summary>
        /// Handles the reset sequence. Returns true if reset is in progress (skip normal step processing).
        /// </summary>
        public bool HandleResetSequence(HeroController hero, HealthManager boss)
        {
            switch (CurrentState)
            {
                case EpisodeState.HeroDead:
                    return HandleReloadReset(hero, boss, "Hero died", HeroDeathResetDelayDuration);

                case EpisodeState.BossDead:
                    return HandleReloadReset(hero, boss, "Boss defeated", DefaultResetDelayDuration);

                case EpisodeState.HeroStuck:
                    return HandleReloadReset(hero, boss, "Hero stuck", DefaultResetDelayDuration);

                default:
                    return false;
            }
        }

        /// <summary>
        /// Resets the episode manager state after a successful reset.
        /// </summary>
        public void ResetEpisode()
        {
            CurrentState = EpisodeState.Training;
            hasTriggeredReset = false;
            hasPressedF5 = false;
            bossSeenThisEpisode = false;
            consecutiveStuckSteps = 0;
            f5PressTime = 0f;
            f5PressAttempts = 0;
            
            RLManager.StaticLogger?.LogInfo("[TrainingEpisodeManager] Episode reset complete, resuming training");
            
            OnResetComplete?.Invoke();
        }

        private bool HandleReloadReset(HeroController hero, HealthManager boss, string reason, float resetDelayDuration)
        {
            if (!hasTriggeredReset)
            {
                hasTriggeredReset = true;
                resetSequenceStartTime = Time.unscaledTime;
                RLManager.StaticLogger?.LogInfo($"[TrainingEpisodeManager] {reason} - starting automatic reset sequence...");
            }

            float now = Time.unscaledTime;
            bool canStartReload = now - resetSequenceStartTime >= resetDelayDuration;
            bool shouldPressF5 =
                canStartReload &&
                (!hasPressedF5 || now - f5PressTime >= f5RetryInterval) &&
                !IsReloadComplete(hero, boss);

            if (shouldPressF5)
            {
                OnSimulateKeyPress?.Invoke(KeyCode.F5);
                hasPressedF5 = true;
                f5PressTime = now;
                f5PressAttempts++;
                RLManager.StaticLogger?.LogInfo($"[TrainingEpisodeManager] F5 pressed (attempt {f5PressAttempts}), waiting for save state reload...");
            }

            if (hasPressedF5 && now - f5PressTime >= ResetSettleDuration && IsReloadComplete(hero, boss))
            {
                RLManager.StaticLogger?.LogInfo("[TrainingEpisodeManager] Save state reload complete");
                ResetEpisode();
                return false;
            }

            return true;
        }

        private bool IsReloadComplete(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            if (IsHeroDead(hero))
                return false;

            return boss.hp > 0;
        }

        private bool IsHeroDead(HeroController hero)
        {
            if (hero == null || hero.playerData == null)
                return false;

            return hero.playerData.health <= 0 ||
                   hero.cState.dead;
        }
    }
    //这个类强依赖 DebugMod 的 Quickslot Load
    //在明确DebugMod 的 相关设置前提下才能保证正确更改这里的逻辑
}
