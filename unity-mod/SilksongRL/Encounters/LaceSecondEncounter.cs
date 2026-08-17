using System;
using System.Collections.Generic;
using UnityEngine;

namespace SilksongRL
{
    /// <summary>
    /// Boss encounter configuration for Lace 2.
    /// </summary>
    public class LaceSecondEncounter : BossEncounterBase
    {
        private const int LaceFsmOneHotSize = 24;
        private const int LaceFsmSemanticSize = 16;
        private const int LaceFsmElapsedSize = 1;
        private const int LacePhaseSize = 3;
        private const int LaceFsmObservationSize = LaceFsmOneHotSize + LaceFsmSemanticSize + LaceFsmElapsedSize;
        private const int LaceBossSpecificObservationSize = LaceFsmObservationSize + LacePhaseSize;
        private const int LaceFsmStateSlots = LaceFsmOneHotSize - 1;
        private const float LaceFsmElapsedNormalizationSeconds = 2f;

        private static readonly string[] PriorityPrimaryStates =
        {
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
        };

        private HealthManager cachedBoss;
        private PlayMakerFSM[] cachedFsms;
        private PlayMakerFSM primaryFsm;
        private Dictionary<string, int> primaryStateIndices;
        private string lastPrimaryState;
        private string trackedPrimaryState;
        private float primaryStateEnteredTime;
        private int currentPhase = 1;

        protected override string BossName => "Lace Boss2 New";
        protected override float MinPosX => 33f;
        protected override float MaxPosX => 77f;
        protected override float MinPosY => 96f;
        protected override float MaxPosY => 110f;
        protected override float MaxBossHP => 800f;
        protected override float MaxBossVelocity => 70f;
        protected override float? LowYWarningThreshold => 100f;
        protected override float? SafeMinPosX => 40f;
        protected override float? SafeMaxPosX => 68f;
        protected override float SafePlatformMaskMargin => 1.6f;
        protected override float SafePlatformWarningMargin => 3.8f;
        protected override float SafePlatformPressureBossDistance => 7.5f;
        protected override float SafePlatformPressureCenteringRewardScale => 3f;
        protected override float SafePlatformPressureMoveInwardReward => 1.2f;
        protected override float SafePlatformPressureMoveOutwardPenalty => 4f;
        protected override float SafePlatformPressureNoMovePenalty => 0.75f;
        protected override float AttackCreditWindowSeconds => 0.25f;
        protected override float AttackCreditHitReward => 100f;
        protected override float AttackSilkGainReward => 250f;
        protected override float EmptyAttackPenalty => 0f;
        protected override float AttackPressReward => 1.25f;
        protected override float AttackOpportunityDistance => 3.0f;
        protected override float AttackOpportunityReward => 5f;
        protected override float AttackRangeMin => 0.9f;
        protected override float AttackRangeMax => 3.0f;
        protected override float AttackVerticalTolerance => 0.75f;
        protected override float VerticalAttackHorizontalTolerance => 1.5f;
        protected override float VerticalAttackRangeMin => 0.5f;
        protected override float VerticalAttackRangeMax => 3.2f;
        protected override float MissedAttackOpportunityPenalty => 0.05f;
        protected override float AttackOpportunityToolPenalty => 25f;
        protected override float CollisionDangerDistance => 0.85f;//本来0.65
        protected override float CollisionDangerPenalty => 8f;
        protected override float CollisionDamagePenalty => 350f;
        protected override float CollisionDangerRetreatRewardScale => 8f;
        protected override float CollisionTacticalDistance => 0.75f;
        protected override float BossDangerDistance => 4.2f;
        protected override float BossDangerVerticalTolerance => 2.2f;
        protected override float BossDangerGreedyActionPenalty => 20f;
        protected override float BossDangerGreedyDamagePenalty => 350f;
        protected override float BossDangerDodgeReward => 15f;
        protected override float BossDangerRetreatRewardScale => 5f;
        protected override float GroundRushDodgeDistance => 5.2f;
        protected override float GroundRushDodgeVerticalTolerance => 1.6f;
        protected override int BossSpecificObservationSize => LaceBossSpecificObservationSize;

        public override bool ShouldForceBasicAttack()
        {
            return true;
        }

        public override bool IsHeroStuck(HeroController hero)
        {
            return hero != null && hero.transform.position.y < 99f;
        }

        public override void ResetObservationHistory()
        {
            base.ResetObservationHistory();
            lastPrimaryState = null;
            trackedPrimaryState = null;
            primaryStateEnteredTime = Time.time;
            currentPhase = 1;
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
                WritePhaseObservations(observations, offset);
                return;
            }

            string stateName = NormalizeStateName(primaryFsm.ActiveStateName);
            lastPrimaryState = stateName;
            UpdatePrimaryStateTimer(stateName);
            UpdatePhaseFromState(stateName);
            int stateIndex;
            if (!primaryStateIndices.TryGetValue(stateName, out stateIndex) || stateIndex >= LaceFsmStateSlots)
            {
                observations[offset] = 1f;
                WriteSemanticFlags(observations, offset, stateName, hero, boss);
                WritePrimaryStateElapsed(observations, offset);
                WritePhaseObservations(observations, offset);
                return;
            }

            observations[offset + 1 + stateIndex] = 1f;
            WriteSemanticFlags(observations, offset, stateName, hero, boss);
            WritePrimaryStateElapsed(observations, offset);
            WritePhaseObservations(observations, offset);
        }

        protected override int GetPhaseFromObservation(float[] observations)
        {
            if (observations == null)
                return 1;

            int start = BaseObservationSize + LaceFsmObservationSize;
            if (observations.Length <= start + 2)
                return 1;

            if (observations[start + 2] > 0.5f)
                return 3;
            if (observations[start + 1] > 0.5f)
                return 2;
            return 1;
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
            currentPhase = 1;
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
                if (state == null)
                    continue;

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

                result[stateName] = index;
                index++;
            }

            foreach (string stateName in stateNames)
            {
                if (index >= LaceFsmStateSlots)
                    break;

                if (result.ContainsKey(stateName))
                    continue;

                result[stateName] = index;
                index++;
            }

            return result;
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

        private string GetCurrentPrimaryStateLower()
        {
            string stateName = primaryFsm != null
                ? NormalizeStateName(primaryFsm.ActiveStateName)
                : NormalizeStateName(lastPrimaryState);
            return stateName.ToLowerInvariant();
        }

        private void WriteSemanticFlags(float[] observations, int offset, string stateName, HeroController hero, HealthManager boss)
        {
            int start = offset + LaceFsmOneHotSize;
            string lowerState = NormalizeStateName(stateName).ToLowerInvariant();

            bool attackOpportunity = IsBossAttackOpportunityState(lowerState);
            bool safeForBind = IsBossSafeForBindState(lowerState);
            bool dangerous = IsBossDangerousState(lowerState);
            bool groundRush = IsBossGroundRushState(lowerState);

            float relX = 0f;
            float relY = 0f;
            float absX = 999f;
            float absY = 999f;
            float distance = 999f;
            bool facingBoss = false;
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

            bool startup = ContainsAny(
                lowerState,
                "antic",
                "aim",
                "break",
                "stance",
                "tele in",
                "tele out",
                "roarantic",
                "crossup");
            bool recovery = ContainsAny(
                lowerState,
                "recover",
                "land",
                "bounce",
                "slash end",
                "rapidslash end",
                "end");
            bool stun = ContainsAny(lowerState, "stun", "stunned");
            bool counterOrParry = ContainsAny(lowerState, "counter", "parry");
            bool teleportOrPhase = ContainsAny(lowerState, "tele", "shift", "roar", "pose");
            bool activeDamage = dangerous &&
                !startup &&
                !recovery &&
                !stun &&
                ContainsAny(
                    lowerState,
                    "slash",
                    "strike",
                    "hit",
                    "downstab",
                    "charge",
                    "rapid",
                    "multi",
                    "slam",
                    "crossslash");

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
            observations[start + 15] = teleportOrPhase ? 1f : 0f;
        }

        private void UpdatePhaseFromState(string stateName)
        {
            string lowerState = NormalizeStateName(stateName).ToLowerInvariant();
            int detectedPhase = currentPhase;
            if (ContainsAny(lowerState, "p3"))
            {
                detectedPhase = 3;
            }
            else if (ContainsAny(lowerState, "p2"))
            {
                detectedPhase = 2;
            }

            detectedPhase = Mathf.Clamp(detectedPhase, 1, 3);
            if (detectedPhase > currentPhase)
            {
                currentPhase = detectedPhase;
                RLManager.StaticLogger?.LogInfo($"[Lace2Phase] Phase advanced to {currentPhase} from primary state '{stateName}'");
            }
        }

        private void WritePhaseObservations(float[] observations, int offset)
        {
            int start = offset + LaceFsmObservationSize;
            int phaseIndex = Mathf.Clamp(currentPhase, 1, 3) - 1;
            observations[start + phaseIndex] = 1f;
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

        private bool IsBossAttackOpportunityState(string lowerState)
        {
            return ContainsAny(
                lowerState,
                "recover",
                "stun",
                "stunned",
                "land",
                "bounce back",
                "slash end");
        }

        private bool IsBossSafeForBindState(string lowerState)
        {
            return ContainsAny(
                lowerState,
                "idle",
                "recover",
                "stun",
                "stunned",
                "land",
                "slash end");
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
                "throw",
                "sphere",
                "needle",
                "parry",
                "counter",
                "strike",
                "hit",
                "slam");
        }

        private bool IsBossGroundRushState(string lowerState)
        {
            if (IsBossAttackOpportunityState(lowerState))
                return false;

            if (ContainsAny(lowerState, "dash", "lunge", "stab", "charge"))
                return true;

            if (ContainsAny(
                    lowerState,
                    "comboslash",
                    "combo slash",
                    "combo strike",
                    "quick slash",
                    "b slash",
                    "multicharge",
                    "multi charge",
                    "multihit slash",
                    "multihitting",
                    "rapidslash charge",
                    "rapidslash loop",
                    "rapid slash charge",
                    "rapid slash loop"))
            {
                return true;
            }

            return
                (ContainsAny(lowerState, "step", "run", "walk") &&
                    ContainsAny(lowerState, "slash", "attack", "combo")) ||
                ContainsAny(lowerState, "slash step", "step slash", "multi slash");
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
