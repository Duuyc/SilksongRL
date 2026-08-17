using System;
using System.Collections.Generic;
using UnityEngine;

namespace SilksongRL
{
    /// <summary>
    /// Boss encounter configuration for Lace 1.
    /// </summary>
    public class LaceEncounter : BossEncounterBase
    {
        private const int LaceFsmOneHotSize = 48;
        private const int LaceFsmSemanticSize = 16;
        private const int LaceFsmElapsedSize = 1;
        private const int LaceFsmObservationSize = LaceFsmOneHotSize + LaceFsmSemanticSize + LaceFsmElapsedSize;
        private const int LaceFsmStateSlots = LaceFsmOneHotSize - 1;
        private const float LaceFsmElapsedNormalizationSeconds = 2f;

        private static readonly string[] PriorityPrimaryStates =
        {
            "Idle",
            "Charge Antic",
            "Charge Break",
            "Charge",
            "Charge Recover",
            "ComboSlash 1",
            "ComboSlash 2",
            "ComboSlash 3",
            "ComboSlash 4",
            "ComboSlash 5",
            "Counter Antic",
            "Counter Stance",
            "Counter Hit",
            "Counter End",
            "RapidSlash Charge",
            "RapidSlash Loop",
            "RapidSlash End",
            "J Slash Antic",
            "J Slash 1",
            "J Slash 2",
            "J Slash 3",
            "J Slash 4",
            "Downstab Antic",
            "Downstab",
            "Downstab Land",
            "CrossSlash Antic",
            "CrossSlash",
            "Slash Slam",
            "Evade",
            "Evade Recover",
            "Evade Move",
            "Hop Antic",
            "Hop",
            "Hop Recover",
            "Stun Air",
            "Stunned",
            "Stun Recover",
            "Damage Recover",
            "Pose Swish",
            "Pose Swish 2",
            "Tele In",
            "Tele Out",
            "Lava Damage",
            "Start Battle"
        };

        private HealthManager cachedBoss;
        private PlayMakerFSM[] cachedFsms;
        private PlayMakerFSM primaryFsm;
        private Dictionary<string, int> primaryStateIndices;
        private string lastPrimaryState;
        private string trackedPrimaryState;
        private float primaryStateEnteredTime;

        protected override string BossName => "Lace Boss1";
        protected override float MinPosX => 77.5f;
        protected override float MaxPosX => 110.5f;
        protected override float MinPosY => 2f;
        protected override float MaxPosY => 25f;
        protected override float MaxBossHP => 250f;
        protected override float MaxBossVelocity => 70f;
        protected override float? LowYWarningThreshold => null;
        protected override float? LowYDamagePenaltyThreshold => null;
        protected override float? SafeMinPosX => 81.5f;
        protected override float? SafeMaxPosX => 105.5f;
        protected override float SafePlatformMaskMargin => 4f;
        protected override float SafePlatformWarningMargin => 5f;
        protected override float SafePlatformPressureMoveInwardReward => 8f;
        protected override float SafePlatformPressureNoMovePenalty => 3f;
        protected override bool AllowToolAndSkillActions => false;
        protected override float AttackRangeMax => 3.3f;
        protected override float AttackVerticalTolerance => 1.0f;
        protected override float VerticalAttackHorizontalTolerance => 1.5f;
        protected override float AttackCreditHitReward => 90f;
        protected override float AttackSilkGainReward => 57.75f;
        protected override float AttackOpportunityReward => 12f;
        protected override float MissedAttackOpportunityPenalty => 17.5f;
        protected override float EmptyAttackPenalty => 0.15f;
        protected override float CollisionDangerDistance => 0.85f;
        protected override float CollisionTacticalDistance => 0.75f;
        protected override float BossDangerDistance => 4.0f;
        protected override float BossDangerVerticalTolerance => 2.1f;
        protected override float GroundRushDodgeDistance => 4.8f;
        protected override float GroundRushDodgeVerticalTolerance => 1.6f;
        protected override int BossSpecificObservationSize => LaceFsmObservationSize;

        public override bool ShouldForceBasicAttack()
        {
            return false;
        }

        public override void ResetObservationHistory()
        {
            base.ResetObservationHistory();
            lastPrimaryState = null;
            trackedPrimaryState = null;
            primaryStateEnteredTime = Time.time;
        }

        protected override void WriteBossSpecificObservations(float[] observations, int offset, HeroController hero, HealthManager boss)
        {
            EnsureFsmCache(boss);

            if (primaryFsm == null || primaryStateIndices == null)
            {
                UpdatePrimaryStateTimer("<none>");
                observations[offset] = 1f;
                WriteSemanticFlags(observations, offset, "<none>", hero, boss);
                WritePrimaryStateElapsed(observations, offset);
                return;
            }

            string stateName = NormalizeStateName(primaryFsm.ActiveStateName);
            lastPrimaryState = stateName;
            UpdatePrimaryStateTimer(stateName);

            int stateIndex;
            if (!primaryStateIndices.TryGetValue(stateName, out stateIndex) || stateIndex >= LaceFsmStateSlots)
            {
                observations[offset] = 1f;
                WriteSemanticFlags(observations, offset, stateName, hero, boss);
                WritePrimaryStateElapsed(observations, offset);
                return;
            }

            observations[offset + 1 + stateIndex] = 1f;
            WriteSemanticFlags(observations, offset, stateName, hero, boss);
            WritePrimaryStateElapsed(observations, offset);
        }

        protected override bool IsBossAttackOpportunityState()
        {
            return IsBossAttackOpportunityState(GetCurrentPrimaryStateLower());
        }

        protected override bool IsBossSafeForBindState()
        {
            return IsBossSafeForBindState(GetCurrentPrimaryStateLower());
        }

        protected override bool IsBossDangerousState()
        {
            return IsBossDangerousState(GetCurrentPrimaryStateLower());
        }

        protected override bool IsBossGroundRushState()
        {
            return IsBossGroundRushState(GetCurrentPrimaryStateLower());
        }

        private void EnsureFsmCache(HealthManager boss)
        {
            if (boss == null)
                return;

            if (cachedBoss == boss && cachedFsms != null)
                return;

            cachedBoss = boss;
            cachedFsms = boss.GetComponentsInChildren<PlayMakerFSM>(true);
            primaryFsm = SelectPrimaryFsm(cachedFsms);
            primaryStateIndices = BuildStateIndex(primaryFsm);
            lastPrimaryState = null;
            trackedPrimaryState = null;
            primaryStateEnteredTime = Time.time;
        }

        private PlayMakerFSM SelectPrimaryFsm(PlayMakerFSM[] fsms)
        {
            if (fsms == null || fsms.Length == 0)
                return null;

            PlayMakerFSM best = null;
            int bestScore = int.MinValue;
            foreach (PlayMakerFSM fsm in fsms)
            {
                if (fsm == null)
                    continue;

                int score = ScoreFsm(fsm);
                if (score > bestScore)
                {
                    best = fsm;
                    bestScore = score;
                }
            }

            return best;
        }

        private int ScoreFsm(PlayMakerFSM fsm)
        {
            string name = (SafeName(fsm.FsmName) + " " + SafeName(fsm.name)).ToLowerInvariant();
            int stateCount = fsm.FsmStates != null ? fsm.FsmStates.Length : 0;
            int score = Mathf.Min(stateCount, 100);

            score += ContainsAny(name, "lace") ? 60 : 0;
            score += ContainsAny(name, "boss") ? 40 : 0;
            score += ContainsAny(name, "control", "main", "attack", "phase", "combat") ? 25 : 0;
            score -= ContainsAny(name, "audio", "sound", "music", "fx", "effect", "corpse", "title", "camera") ? 50 : 0;

            if (fsm.Active)
                score += 10;

            return score;
        }

        private Dictionary<string, int> BuildStateIndex(PlayMakerFSM fsm)
        {
            Dictionary<string, int> result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            if (fsm == null || fsm.FsmStates == null)
                return result;

            SortedSet<string> stateNames = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var state in fsm.FsmStates)
            {
                if (state != null)
                    stateNames.Add(NormalizeStateName(state.Name));
            }

            int index = 0;
            foreach (string priorityState in PriorityPrimaryStates)
            {
                if (index >= LaceFsmStateSlots)
                    break;

                string stateName = NormalizeStateName(priorityState);
                if (!stateNames.Contains(stateName) || result.ContainsKey(stateName))
                    continue;

                result[stateName] = index++;
            }

            foreach (string stateName in stateNames)
            {
                if (index >= LaceFsmStateSlots)
                    break;

                if (!result.ContainsKey(stateName))
                    result[stateName] = index++;
            }

            return result;
        }

        private void WriteSemanticFlags(float[] observations, int offset, string stateName, HeroController hero, HealthManager boss)
        {
            int start = offset + LaceFsmOneHotSize;
            string lowerState = NormalizeStateName(stateName).ToLowerInvariant();

            bool attackOpportunity = IsBossAttackOpportunityState(lowerState);
            bool safeForBind = IsBossSafeForBindState(lowerState);
            bool dangerous = IsBossDangerousState(lowerState);
            bool groundRush = IsBossGroundRushState(lowerState);

            float relX;
            float relY;
            float absX;
            float absY;
            float distance;
            bool facingBoss;
            bool hasGeometry = TryGetGeometry(hero, boss, out relX, out relY, out absX, out absY, out distance, out facingBoss);

            bool bossAboveHero = hasGeometry && relY > 1.0f;
            bool nearCollision = hasGeometry &&
                CollisionTacticalDistance > 0f &&
                distance <= CollisionTacticalDistance;
            bool nearDangerReach = hasGeometry &&
                dangerous &&
                absX <= BossDangerDistance &&
                absY <= BossDangerVerticalTolerance;
            bool flatSlashWindow = hasGeometry &&
                attackOpportunity &&
                facingBoss &&
                absX >= AttackRangeMin &&
                absX <= AttackRangeMax &&
                absY <= AttackVerticalTolerance;
            bool verticalSlashWindow = hasGeometry &&
                attackOpportunity &&
                relY >= VerticalAttackRangeMin &&
                relY <= VerticalAttackRangeMax &&
                absX <= VerticalAttackHorizontalTolerance;
            bool bindReady = PlayerData.HasInstance &&
                hero != null &&
                hero.playerData != null &&
                hero.playerData.health < Mathf.Max(1, hero.playerData.CurrentMaxHealth) &&
                PlayerData.instance.silk >= 9;
            bool safeBindWindow = bindReady &&
                (safeForBind || (hasGeometry && distance >= SafeBindFarDistanceThreshold));

            bool startup = ContainsAny(lowerState, "antic", "aim", "break", "stance", "wind", "tele in", "rapidslash charge");
            bool recovery = ContainsAny(lowerState, "recover", "land", "bounce", "slash end", "end", "pose swish");
            bool stun = ContainsAny(lowerState, "stun", "stunned");
            bool counterOrParry = ContainsAny(lowerState, "counter", "parry");
            bool airborneOrTeleport = ContainsAny(lowerState, "air", "jump", "hop", "tele", "j slash", "downstab");
            bool activeDamage = dangerous &&
                !startup &&
                !recovery &&
                !stun &&
                ContainsAny(lowerState, "slash", "strike", "hit", "stab", "charge", "circle", "lunge", "dash", "slam");

            observations[start + 0] = dangerous ? 1f : 0f;
            observations[start + 1] = attackOpportunity ? 1f : 0f;
            observations[start + 2] = safeForBind ? 1f : 0f;
            observations[start + 3] = groundRush ? 1f : 0f;
            observations[start + 4] = bossAboveHero ? 1f : 0f;
            observations[start + 5] = nearCollision ? 1f : 0f;
            observations[start + 6] = nearDangerReach ? 1f : 0f;
            observations[start + 7] = flatSlashWindow ? 1f : 0f;
            observations[start + 8] = verticalSlashWindow ? 1f : 0f;
            observations[start + 9] = safeBindWindow ? 1f : 0f;
            observations[start + 10] = startup ? 1f : 0f;
            observations[start + 11] = activeDamage ? 1f : 0f;
            observations[start + 12] = recovery ? 1f : 0f;
            observations[start + 13] = stun ? 1f : 0f;
            observations[start + 14] = counterOrParry ? 1f : 0f;
            observations[start + 15] = airborneOrTeleport ? 1f : 0f;
        }

        private bool TryGetGeometry(
            HeroController hero,
            HealthManager boss,
            out float relX,
            out float relY,
            out float absX,
            out float absY,
            out float distance,
            out bool facingBoss)
        {
            relX = 0f;
            relY = 0f;
            absX = 0f;
            absY = 0f;
            distance = 0f;
            facingBoss = false;

            if (hero == null || boss == null)
                return false;

            Vector2 heroPos = hero.transform.position;
            Vector2 bossPos = boss.transform.position;
            relX = bossPos.x - heroPos.x;
            relY = bossPos.y - heroPos.y;
            absX = Mathf.Abs(relX);
            absY = Mathf.Abs(relY);
            distance = Vector2.Distance(heroPos, bossPos);

            bool bossIsRight = relX >= 0f;
            bool heroFacingRight = hero.cState.facingRight;
            facingBoss = bossIsRight ? heroFacingRight : !heroFacingRight;
            return true;
        }

        private string GetCurrentPrimaryStateLower()
        {
            string stateName = primaryFsm != null
                ? NormalizeStateName(primaryFsm.ActiveStateName)
                : NormalizeStateName(lastPrimaryState);
            return stateName.ToLowerInvariant();
        }

        private bool IsBossAttackOpportunityState(string lowerState)
        {
            return ContainsAny(lowerState, "idle", "recover", "stun", "stunned", "land", "bounce back", "slash end", "end", "pose swish");
        }

        private bool IsBossSafeForBindState(string lowerState)
        {
            return ContainsAny(lowerState, "idle", "recover", "stun", "stunned", "land", "slash end", "end", "pose swish");
        }

        private bool IsBossDangerousState(string lowerState)
        {
            if (IsBossAttackOpportunityState(lowerState))
                return false;

            return ContainsAny(
                lowerState,
                "attack",
                "slash",
                "dash",
                "lunge",
                "stab",
                "charge",
                "strike",
                "hit",
                "circle",
                "downstab",
                "counter",
                "slam");
        }

        private bool IsBossGroundRushState(string lowerState)
        {
            if (IsBossAttackOpportunityState(lowerState))
                return false;

            return ContainsAny(lowerState, "charge", "rapidslash loop", "rapid slash loop", "downstab") ||
                ContainsAny(lowerState, "comboslash", "combo slash") ||
                (ContainsAny(lowerState, "run", "walk", "step", "dash", "lunge") &&
                    ContainsAny(lowerState, "slash", "attack", "strike", "stab"));
        }

        private void UpdatePrimaryStateTimer(string stateName)
        {
            stateName = NormalizeStateName(stateName);
            if (trackedPrimaryState == stateName)
                return;

            trackedPrimaryState = stateName;
            primaryStateEnteredTime = Time.time;
        }

        private void WritePrimaryStateElapsed(float[] observations, int offset)
        {
            float elapsed = Mathf.Max(0f, Time.time - primaryStateEnteredTime);
            observations[offset + LaceFsmOneHotSize + LaceFsmSemanticSize] =
                Mathf.Clamp01(elapsed / LaceFsmElapsedNormalizationSeconds);
        }

        private bool ContainsAny(string value, params string[] tokens)
        {
            foreach (string token in tokens)
            {
                if (value.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }

            return false;
        }

        private string NormalizeStateName(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? "<none>" : value.Trim();
        }

        private string SafeName(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? "<unnamed>" : value.Trim();
        }
    }
}
