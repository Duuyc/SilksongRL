using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace SilksongRL
{
    public abstract class BossEncounterBase : IBossEncounter
    {
        protected const int CoreObservationSize = 24;
        protected const int ResourceObservationSize = 19;
        protected const int RaycastCount = 32;
        protected const int RaycastHitTypeCount = 6;
        protected const int RaycastObservationSize = RaycastCount + RaycastCount * RaycastHitTypeCount;
        protected const int HitboxObservationCount = 12;
        protected const int HitboxTypeCount = 6;
        protected const int HitboxFeatureSize = 1 + 4 + HitboxTypeCount;
        protected const int HitboxObservationSize = HitboxObservationCount * HitboxFeatureSize;
        protected const int BaseObservationSize = CoreObservationSize + ResourceObservationSize + RaycastObservationSize + HitboxObservationSize;
        protected const int FrameStackSize = 16;
        private static readonly int[] FrameSampleOffsets = { 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23 };
        private const int FrameHistorySize = 24;

        protected const int HeroHpIndex = 4;
        protected const int BossHpIndex = 9;
        protected const int DistanceIndex = 14;
        protected const int SilkIndex = CoreObservationSize;
        protected const int ToolSlotStartIndex = CoreObservationSize + 4;
        protected const int ToolSlotSize = 5;
        protected const int HeroFacingIndex = 21;
        protected const int HeroFacingBossIndex = 22;
        protected const int BossFacingIndex = 23;

        private const float MaxRaycastDistance = 25f;
        private const float LaceCircleSlashVirtualRadius = 3.0f;
        private const float LaceCrossSlashVirtualRadius = 5.0f;
        private const float MaxHitboxObservedWidth = 12f;
        private const float MaxHitboxObservedHeight = 12f;
        private const float MaxObservedSilk = 18f;
        private const float MaxObservedToolCapacity = 20f;
        private const float MaxObservedSkillCost = 9f;
        private const float BoundaryMaskMargin = 1.0f;
        private const float RewardReferenceStepSeconds = 0.025f;

        private float[] stackedObservationHistory;
        private int stackedSingleFrameSize;
        private HealthManager stackedBoss;
        private string stackedSceneName;
        private float pendingAttackCreditSeconds;
        private float pendingDodgeOverBossDamageWindowSeconds;
        private float lastRewardTime = -1f;
        private RewardComponents lastRewardComponents = new RewardComponents();

        protected abstract string BossName { get; }
        protected abstract float MinPosX { get; }
        protected abstract float MaxPosX { get; }
        protected abstract float MinPosY { get; }
        protected abstract float MaxPosY { get; }
        protected abstract float MaxBossHP { get; }

        protected virtual float MaxHeroVelocity => 30f;
        protected virtual float MaxBossVelocity => 70f;
        protected virtual float MaxHeroHP => 10f;
        protected virtual float FarDistancePenaltyThreshold => 15f;
        protected virtual float? LowYWarningThreshold => MinPosY + (MaxPosY - MinPosY) * 0.2f;
        protected virtual float? LowYDamagePenaltyThreshold => MinPosY + (MaxPosY - MinPosY) * 0.15f;
        protected virtual float? SafeMinPosX => null;
        protected virtual float? SafeMaxPosX => null;
        protected virtual float SafePlatformMaskMargin => 2f;
        protected virtual float SafePlatformWarningMargin => 4f;
        protected virtual float SafePlatformOutsidePenalty => 8f;
        protected virtual float SafePlatformEdgePenalty => 1.5f;
        protected virtual float SafePlatformCenteringRewardScale => 2f;
        protected virtual float SafePlatformDamagePenalty => 500f;
        protected virtual float SafePlatformPressureBossDistance => 8f;
        protected virtual float SafePlatformPressureCenteringRewardScale => 8f;
        protected virtual float SafePlatformPressureMoveInwardReward => 4f;
        protected virtual float SafePlatformPressureMoveOutwardPenalty => 4f;
        protected virtual float SafePlatformPressureNoMovePenalty => 1.5f;
        protected virtual float AttackCreditWindowSeconds => 0.25f;
        protected virtual float AttackCreditHitReward => 25f;
        protected virtual float AttackSilkGainReward => 20f;
        protected virtual float EmptyAttackPenalty => 0.25f;
        protected virtual float AttackPressReward => 0f;
        protected virtual float AttackOpportunityDistance => 3.3f;//本来是5
        protected virtual float AttackOpportunityReward => 2f;
        protected virtual float AttackRangeMin => 0.8f;
        protected virtual float AttackRangeMax => AttackOpportunityDistance;
        protected virtual float AttackVerticalTolerance => 1.2f;
        protected virtual float VerticalAttackHorizontalTolerance => 1.5f;
        protected virtual float VerticalAttackRangeMin => 0.5f;
        protected virtual float VerticalAttackRangeMax => AttackRangeMax;
        protected virtual float AntiAirPriorityMinRelY => VerticalAttackRangeMin;
        protected virtual float SafeCloseSlashMaskMinDistance => 0.65f;
        protected virtual float SafeCloseSlashMaskMaxDistance => AttackRangeMax;
        protected virtual float MissedAttackOpportunityPenalty => 0f;
        protected virtual float AttackOpportunityToolPenalty => 0f;
        protected virtual float ToolAmmoUsePenalty => -30f;
        protected virtual float CollisionDangerDistance => 0f;
        protected virtual float CollisionDangerPenalty => 0f;
        protected virtual float CollisionDamagePenalty => 0f;
        protected virtual float CollisionDangerRetreatRewardScale => 0f;
        protected virtual float CollisionTacticalDistance => CollisionDangerDistance;
        protected virtual float CollisionTacticalVerticalTolerance => AttackVerticalTolerance;
        protected virtual float BossDangerDistance => 0f;
        protected virtual float BossDangerVerticalTolerance => AttackVerticalTolerance;
        protected virtual float BossDangerGreedyActionPenalty => 0f;
        protected virtual float BossDangerGreedyDamagePenalty => 0f;
        protected virtual float BossDangerDodgeReward => 0f;
        protected virtual float BossDangerRetreatRewardScale => 0f;
        protected virtual float GroundRushDodgeDistance => BossDangerDistance;
        protected virtual float GroundRushDodgeVerticalTolerance => BossDangerVerticalTolerance;
        protected virtual float DodgeOverBossMinGroundRushDistance => Mathf.Max(2.4f, CollisionDangerDistance + 1.2f);
        protected virtual float DodgeOverBossEdgeClearance => SafePlatformMaskMargin + 0.7f;
        protected virtual float DodgeOverBossDamageWindowSeconds => 0.45f;
        protected virtual float DodgeOverBossDamagePenalty => 600f;
        protected virtual float PredictiveDodgeSuccessReward => 100f;
        protected virtual int PredictiveDodgeNewestHistoryFrame => 8;
        protected virtual int PredictiveDodgeOldestHistoryFrame => FrameStackSize - 1;
        protected virtual float PredictiveDodgeStepSeconds => 0.05f;
        protected virtual float PredictiveDodgeAttackLeadDistance => 1f;
        protected virtual float PredictiveDodgeHitRadius => 0.75f;
        protected virtual float PredictiveDodgeSafeExitMargin => 0.15f;
        protected virtual float PredictiveDodgeVelocityEscapeDistance => 3.5f;
        protected virtual float PredictiveDodgeVelocityEscapeExitMargin => 0.25f;
        protected virtual float PredictiveDodgeMinBossMotion => 1f;
        protected virtual float PredictiveDodgeMinHeroMotion => 0.2f;
        protected virtual float PredictiveDodgeMaxAcceleration => 250f;
        protected virtual float BindAttemptReward => AttackCreditHitReward + AttackSilkGainReward;
        protected virtual float HighSilkHoldLowHpMultiplier => 0.1f;
        protected virtual float HighSilkHoldHighHpMultiplier => 1.0f;
        protected virtual float HighSilkHoldLowHpPercent => 0.1f;
        protected virtual float HighSilkHoldHighHpPercent => 0.9f;
        protected virtual float LowHealthBindResourceMaskHpPercent => 0.4f;
        protected virtual float SafeBindFarDistanceThreshold => 9f;
        protected virtual float LowHealthHealMultiplier => 2;
        protected virtual float HighHealthHealMultiplier => 1f;
        protected virtual float HealMultiplierLowHpPercent => 0.1f;
        protected virtual float HealMultiplierHighHpPercent => 0.9f;
        protected virtual float LowHealthDamagePenaltyMultiplier => 1.5f;
        protected virtual float HighHealthDamagePenaltyMultiplier => 1.0f;
        protected virtual float DamagePenaltyLowHpPercent => 0.1f;
        protected virtual float DamagePenaltyHighHpPercent => 0.9f;
        protected virtual float PhaseProgressReward => 1500f;
        protected virtual float PhaseRewardMultiplierBase => 1.2f;
        protected virtual float TacticalDangerDistance => BossDangerDistance;
        protected virtual float TacticalDangerVerticalTolerance => BossDangerVerticalTolerance;
        protected virtual bool AllowToolAndSkillActions => true;
        protected virtual int BossSpecificObservationSize => 0;

        protected int SingleFrameObservationSize => BaseObservationSize + BossSpecificObservationSize;

        public string GetEncounterName()
        {
            return BossName;
        }

        public ObservationType GetObservationType()
        {
            return ObservationType.Vector;
        }

        public int GetVectorObservationSize()
        {
            return GetObservationSize();
        }

        public (int width, int height) GetVisualObservationSize()
        {
            return (0, 0);
        }

        public virtual bool IsEncounterMatch(HealthManager hm)
        {
            return hm != null && hm.name == BossName;
        }

        public virtual float[] ExtractObservationArray(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return null;

            float[] currentFrame = new float[SingleFrameObservationSize];
            WriteCoreObservations(currentFrame, hero, boss);
            WriteResourceObservations(currentFrame);
            WriteRaycastObservations(currentFrame, hero, boss);
            WriteHitboxObservations(currentFrame, hero, boss);
            WriteBossSpecificObservations(currentFrame, BaseObservationSize, hero, boss);
            return BuildStackedObservation(currentFrame, boss);
        }

        public virtual int[] GetActionMask(HeroController hero, HealthManager boss)
        {
            int[] rawMask = GetRawActionMask(hero, boss);
            if (ActionManager.UsesRawActionSpace())
                return rawMask;

            int[] macroMask = ActionManager.CreateAllValidActionMask();
            int[] tacticalMacroMask = CreateMacroMask();
            ApplyTacticalMacroMask(tacticalMacroMask, hero, boss);
            bool forceBasicAttack = ShouldForceBasicAttack();

            for (int i = 0; i < ActionManager.GetActionCount(); i++)
            {
                if (forceBasicAttack && i >= 0 && i < ActionManager.MovementMacroCount)
                {
                    ActionManager.DisableActionValue(macroMask, ActionManager.MacroOffset, i);
                    continue;
                }

                MacroAction macro = ActionManager.ClampMacroAction(i);
                if (!AllowToolAndSkillActions &&
                    (ActionManager.IsRangedToolAction(macro) || ActionManager.IsCloseSkillAction(macro)))
                {
                    ActionManager.DisableActionValue(macroMask, ActionManager.MacroOffset, i);
                    continue;
                }

                Action rawAction = ActionManager.ActionIndexToAction(i, hero, boss, this);
                if (!ActionManager.IsInternallyConsistent(rawAction) ||
                    !IsRawActionAllowed(rawMask, rawAction) ||
                    !IsMacroEnabled(tacticalMacroMask, rawAction.macro))
                {
                    ActionManager.DisableActionValue(macroMask, ActionManager.MacroOffset, i);
                }
            }

            ActionManager.EnsureValidChoices(macroMask);
            return macroMask;
        }

        private int[] CreateMacroMask()
        {
            int[] mask = new int[ActionManager.MacroActionCount];
            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = 1;
            }

            return mask;
        }

        public virtual float GetPreferredCenterX()
        {
            if (SafeMinPosX.HasValue && SafeMaxPosX.HasValue)
            {
                return (SafeMinPosX.Value + SafeMaxPosX.Value) * 0.5f;
            }

            return (MinPosX + MaxPosX) * 0.5f;
        }

        public virtual bool IsBossDangerous(HeroController hero, HealthManager boss)
        {
            return IsBossDangerousState();
        }

        public virtual MacroAction GetTeacherAction(HeroController hero, HealthManager boss, int[] actionMask)
        {
            MacroAction desired;
            if (hero == null || boss == null)
            {
                desired = MacroAction.HoldSafeSide;
            }
            else if (ShouldUseCollisionTacticalMask(hero, boss))
            {
                desired = MacroAction.RetreatFromBossDodge;
            }
            else if (ShouldUseGroundRushDodgeMask(hero, boss))
            {
                if (IsDodgeOverBossViable(hero, boss))
                {
                    desired = MacroAction.DodgeOverBossHighJump;
                }
                else if (IsBossAboveHero(hero, boss))
                {
                    desired = MacroAction.CrossUnderAirBoss;
                }
                else
                {
                    desired = IsUnderSafePlatformPressure(hero, boss)
                        ? MacroAction.MoveToCenterSafely
                        : MacroAction.RetreatFromBossDodge;
                }
            }
            else if (ShouldUseDangerTacticalMask(hero, boss))
            {
                desired = IsBossAboveHero(hero, boss)
                    ? MacroAction.CrossUnderAirBoss
                    : IsUnderSafePlatformPressure(hero, boss)
                        ? MacroAction.MoveToCenterSafely
                        : MacroAction.RetreatFromBossDodge;
            }
            else if (ShouldPrioritizeBind(hero, boss))
            {
                desired = IsBossSafeForBindState()
                    ? MacroAction.BindHeal
                    : MacroAction.RetreatFromBossDodge;
            }
            else if (TryGetSafeCloseSlashOpportunity(hero, boss, out desired))
            {
            }
            else if (IsUnderSafePlatformPressure(hero, boss))
            {
                desired = MacroAction.MoveToCenterSafely;
            }
            else if (IsBossAboveHero(hero, boss))
            {
                desired = MacroAction.CrossUnderAirBoss;
            }
            else
            {
                desired = GetRangeTeacherAction(hero, boss);
            }

            return FirstEnabledAction(
                actionMask,
                hero,
                boss,
                desired,
                MacroAction.KeepSweetSpot,
                MacroAction.RetreatFromBossDodge,
                MacroAction.MoveToCenterSafely,
                MacroAction.HoldSafeSide);
        }

        public virtual bool ShouldForceBasicAttack()
        {
            return false;
        }

        public virtual MoveDirection ClampMovementToSafeArea(HeroController hero, MoveDirection requestedDirection)
        {
            if (!SafeMinPosX.HasValue || !SafeMaxPosX.HasValue || hero == null)
                return requestedDirection;

            float heroX = hero.transform.position.x;
            float safeMin = SafeMinPosX.Value;
            float safeMax = SafeMaxPosX.Value;
            float warningMargin = Mathf.Max(SafePlatformMaskMargin, SafePlatformWarningMargin);
            float emergencyMargin = Mathf.Max(0.35f, SafePlatformMaskMargin * 0.5f);

            if (heroX <= safeMin)
                return MoveDirection.Right;
            if (heroX >= safeMax)
                return MoveDirection.Left;

            if (requestedDirection == MoveDirection.None && heroX <= safeMin + emergencyMargin)
                return MoveDirection.Right;
            if (requestedDirection == MoveDirection.None && heroX >= safeMax - emergencyMargin)
                return MoveDirection.Left;

            if (heroX <= safeMin + warningMargin && requestedDirection == MoveDirection.Left)
                return MoveDirection.Right;
            if (heroX >= safeMax - warningMargin && requestedDirection == MoveDirection.Right)
                return MoveDirection.Left;

            return requestedDirection;
        }

        private int[] GetRawActionMask(HeroController hero, HealthManager boss)
        {
            int[] mask = ActionManager.CreateRawAllValidActionMask();

            if (hero == null)
            {
                return mask;
            }

            float heroX = hero.transform.position.x;
            float margin = Mathf.Max(BoundaryMaskMargin, ArenaWidth * 0.04f);
            if (heroX <= MinPosX + margin)
            {
                ActionManager.DisableActionValue(mask, ActionManager.MoveOffset, (int)MoveDirection.Left);
            }
            if (heroX >= MaxPosX - margin)
            {
                ActionManager.DisableActionValue(mask, ActionManager.MoveOffset, (int)MoveDirection.Right);
            }
            ApplySafePlatformMask(mask, hero);

            bool recoiling = hero.cState.recoiling;
            if (recoiling)
            {
                ActionManager.DisableActionValue(mask, ActionManager.JumpOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.AttackOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.DashOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.NeedleOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.BindOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Neutral);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Up);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Down);
            }
            else
            {
                bool reserveSilkForBind = ShouldReserveSilkForBind(hero);
                bool protectBindResources = ShouldProtectBindResources(hero, boss);

                if (hero.cState.dashing)
                {
                    ActionManager.DisableActionValue(mask, ActionManager.DashOffset, 1);
                }

                if (reserveSilkForBind || protectBindResources)
                {
                    ActionManager.DisableActionValue(mask, ActionManager.NeedleOffset, 1);
                }

                if (!CanBind(hero))
                {
                    ActionManager.DisableActionValue(mask, ActionManager.BindOffset, 1);
                }

                if (!CanUseToolSlot(AttackToolBinding.Neutral, reserveSilkForBind) ||
                    (protectBindResources && IsSilkSpendingToolSlot(AttackToolBinding.Neutral)))
                {
                    ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Neutral);
                }
                if (!CanUseToolSlot(AttackToolBinding.Up, reserveSilkForBind) ||
                    (protectBindResources && IsSilkSpendingToolSlot(AttackToolBinding.Up)))
                {
                    ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Up);
                }
                if (!CanUseToolSlot(AttackToolBinding.Down, reserveSilkForBind) ||
                    (protectBindResources && IsSilkSpendingToolSlot(AttackToolBinding.Down)))
                {
                    ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Down);
                }
            }

            if (!AllowToolAndSkillActions)
            {
                ActionManager.DisableActionValue(mask, ActionManager.NeedleOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Neutral);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Up);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Down);
            }

            if (ShouldForceBasicAttack() &&
                ActionManager.IsBasicAttackTapActiveOrReady() &&
                ActionManager.IsActionValueEnabled(mask, ActionManager.AttackOffset, 1))
            {
                ActionManager.DisableActionValue(mask, ActionManager.AttackOffset, 0);
                ActionManager.DisableActionValue(mask, ActionManager.NeedleOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.BindOffset, 1);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Neutral);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Up);
                ActionManager.DisableActionValue(mask, ActionManager.ToolOffset, (int)ToolAction.Down);
            }

            ActionManager.EnsureRawValidChoices(mask);
            return mask;
        }

        private bool IsRawActionAllowed(int[] rawMask, Action action)
        {
            if (rawMask == null || action == null)
                return true;

            return
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.MoveOffset, (int)action.move) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.LookOffset, (int)action.look) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.JumpOffset, action.jump ? 1 : 0) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.AttackOffset, action.attack ? 1 : 0) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.DashOffset, action.dash ? 1 : 0) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.NeedleOffset, action.needle ? 1 : 0) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.BindOffset, action.bind ? 1 : 0) &&
                ActionManager.IsActionValueEnabled(rawMask, ActionManager.ToolOffset, (int)action.tool);
        }

        private void ApplyTacticalMacroMask(int[] macroMask, HeroController hero, HealthManager boss)
        {
            if (macroMask == null || hero == null || boss == null)
                return;

            if (ShouldUseCollisionTacticalMask(hero, boss))
            {
                if (IsUnderSafePlatformPressure(hero, boss))
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.MoveToCenterSafely,
                        MacroAction.HoldSafeSide);
                }
                else
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.HoldSafeSide);
                }
                return;
            }

            if (ShouldUseGroundRushDodgeMask(hero, boss))
            {
                bool canJumpOverBoss = IsDodgeOverBossViable(hero, boss);
                bool underPlatformPressure = IsUnderSafePlatformPressure(hero, boss);
                if (canJumpOverBoss && underPlatformPressure)
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.DodgeOverBossHighJump,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.MoveToCenterSafely,
                        MacroAction.UseCloseSkill);
                }
                else if (canJumpOverBoss)
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.DodgeOverBossHighJump,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.CrossUnderAirBoss,
                        MacroAction.UseCloseSkill);
                }
                else if (underPlatformPressure)
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.CrossUnderAirBoss,
                        MacroAction.MoveToCenterSafely,
                        MacroAction.UseCloseSkill,
                        MacroAction.HoldSafeSide);
                }
                else
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.CrossUnderAirBoss,
                        MacroAction.UseCloseSkill,
                        MacroAction.HoldSafeSide);
                }
                return;
            }

            if (ShouldUseDangerTacticalMask(hero, boss))
            {
                bool canJumpOverBoss = IsDodgeOverBossViable(hero, boss);
                bool underPlatformPressure = IsUnderSafePlatformPressure(hero, boss);
                if (canJumpOverBoss && underPlatformPressure)
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.DodgeOverBossHighJump,
                        MacroAction.CrossUnderAirBoss,
                        MacroAction.MoveToCenterSafely);
                }
                else if (canJumpOverBoss)
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.DodgeOverBossHighJump,
                        MacroAction.CrossUnderAirBoss);
                }
                else if (underPlatformPressure)
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.CrossUnderAirBoss,
                        MacroAction.MoveToCenterSafely,
                        MacroAction.HoldSafeSide);
                }
                else
                {
                    AllowOnlyMacroActions(
                        macroMask,
                        MacroAction.RetreatFromBossDodge,
                        MacroAction.CrossUnderAirBoss,
                        MacroAction.HoldSafeSide);
                }
                return;
            }

            if (ShouldPrioritizeBind(hero, boss))
            {
                DisableMacroActions(
                    macroMask,
                    MacroAction.PunishWithSlash,
                    MacroAction.AntiAirSlash,
                    MacroAction.UseCloseSkill,
                    MacroAction.UseRangedTool);
                return;
            }

            if (TryGetSafeCloseSlashOpportunity(hero, boss, out MacroAction slashMacro))
            {
                AllowOnlyMacroActions(macroMask, slashMacro);
                return;
            }

            if (IsFlatAttackOpportunity(hero, boss))
            {
                AllowOnlyMacroActions(macroMask, MacroAction.PunishWithSlash);
            }
        }

        private bool ShouldUseCollisionTacticalMask(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || CollisionTacticalDistance <= 0f)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            return absX <= CollisionTacticalDistance && absY <= CollisionTacticalVerticalTolerance;
        }

        private bool ShouldUseGroundRushDodgeMask(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || !IsBossGroundRushState())
                return false;

            if (GroundRushDodgeDistance <= 0f)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            return absX <= GroundRushDodgeDistance && absY <= GroundRushDodgeVerticalTolerance;
        }

        private bool IsDodgeOverBossViable(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || !IsBossGroundRushState())
                return false;

            float heroX = hero.transform.position.x;
            float bossX = boss.transform.position.x;
            float absX = Mathf.Abs(bossX - heroX);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            float maxDistance = Mathf.Max(DodgeOverBossMinGroundRushDistance, GroundRushDodgeDistance);

            if (absX < DodgeOverBossMinGroundRushDistance || absX > maxDistance)
                return false;

            if (absY > GroundRushDodgeVerticalTolerance)
                return false;

            if (!SafeMinPosX.HasValue || !SafeMaxPosX.HasValue)
                return true;

            float safeMin = SafeMinPosX.Value + DodgeOverBossEdgeClearance;
            float safeMax = SafeMaxPosX.Value - DodgeOverBossEdgeClearance;
            bool bossIsLeft = bossX < heroX;

            if (bossIsLeft && heroX <= safeMin)
                return false;

            if (!bossIsLeft && heroX >= safeMax)
                return false;

            return true;
        }

        private bool ShouldUseDangerTacticalMask(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || !IsBossDangerousState())
                return false;

            if (TacticalDangerDistance <= 0f)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            return absX <= TacticalDangerDistance && absY <= TacticalDangerVerticalTolerance;
        }

        private bool ShouldPrioritizeBind(HeroController hero, HealthManager boss)
        {
            return CanBind(hero) && ShouldProtectBindResources(hero, boss);
        }

        private bool IsFlatAttackOpportunity(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || !IsBossAttackOpportunityState())
                return false;

            float relX = boss.transform.position.x - hero.transform.position.x;
            float absX = Mathf.Abs(relX);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            return IsFacingBoss(hero, boss) &&
                absX >= AttackRangeMin &&
                absX <= AttackRangeMax &&
                absY <= AttackVerticalTolerance;
        }

        private bool TryGetSafeCloseSlashOpportunity(
            HeroController hero,
            HealthManager boss,
            out MacroAction slashMacro)
        {
            slashMacro = MacroAction.PunishWithSlash;
            if (hero == null || boss == null || !IsBossAttackOpportunityState())
                return false;

            if (hero.cState.dead || hero.cState.recoiling || hero.cState.dashing)
                return false;

            float relX = boss.transform.position.x - hero.transform.position.x;
            float relY = boss.transform.position.y - hero.transform.position.y;
            float absX = Mathf.Abs(relX);
            float absY = Mathf.Abs(relY);
            float minDistance = Mathf.Max(SafeCloseSlashMaskMinDistance, CollisionDangerDistance + 0.05f);

            if (IsVerticalAttackOpportunityGeometry(relY, absX, absY))
            {
                slashMacro = MacroAction.AntiAirSlash;
                return true;
            }

            if (absY <= AttackVerticalTolerance &&
                absX >= minDistance &&
                absX <= SafeCloseSlashMaskMaxDistance)
            {
                slashMacro = MacroAction.PunishWithSlash;
                return true;
            }

            return false;
        }

        private bool IsUnderSafePlatformPressure(HeroController hero, HealthManager boss)
        {
            if (!SafeMinPosX.HasValue || !SafeMaxPosX.HasValue || hero == null || boss == null)
                return false;

            float heroX = hero.transform.position.x;
            float bossX = boss.transform.position.x;
            float safeMin = SafeMinPosX.Value;
            float safeMax = SafeMaxPosX.Value;
            float warningMargin = Mathf.Max(0.1f, SafePlatformWarningMargin);
            bool nearEdge = heroX < safeMin + warningMargin || heroX > safeMax - warningMargin;
            return nearEdge && Mathf.Abs(bossX - heroX) <= SafePlatformPressureBossDistance;
        }

        private bool IsBossAboveHero(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            float relY = boss.transform.position.y - hero.transform.position.y;
            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            return relY >= 1.5f && absX <= BossDangerDistance + 1.5f;
        }

        private MacroAction GetRangeTeacherAction(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return MacroAction.HoldSafeSide;

            float relX = boss.transform.position.x - hero.transform.position.x;
            float absX = Mathf.Abs(relX);
            if (absX < AttackRangeMin)
                return MacroAction.RetreatFromBossDodge;
            if (absX > AttackRangeMax)
                return MacroAction.KeepSweetSpot;
            if (!IsFacingBoss(hero, boss))
                return MacroAction.KeepSweetSpot;

            return MacroAction.HoldSafeSide;
        }

        private MacroAction GetFlatAttackTeacherAction(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return MacroAction.PunishWithSlash;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            if (absX > AttackRangeMax)
                return MacroAction.KeepSweetSpot;

            return MacroAction.PunishWithSlash;
        }

        private MacroAction FirstEnabledAction(
            int[] actionMask,
            HeroController hero,
            HealthManager boss,
            params MacroAction[] actions)
        {
            foreach (MacroAction action in actions)
            {
                if (IsMacroEnabledInActionMask(actionMask, action, hero, boss))
                    return action;
            }

            return MacroAction.HoldSafeSide;
        }

        private void AllowOnlyMacroActions(int[] macroMask, params MacroAction[] allowedActions)
        {
            for (int i = 0; i < ActionManager.MacroActionCount; i++)
            {
                MacroAction macro = (MacroAction)i;
                bool allowed = false;
                foreach (MacroAction allowedAction in allowedActions)
                {
                    if (MacroActionMatchesRule(macro, allowedAction))
                    {
                        allowed = true;
                        break;
                    }
                }

                if (!allowed)
                {
                    ActionManager.DisableActionValue(macroMask, ActionManager.MacroOffset, i);
                }
            }
        }

        private void DisableMacroActions(int[] macroMask, params MacroAction[] disabledActions)
        {
            for (int i = 0; i < ActionManager.MacroActionCount; i++)
            {
                MacroAction macro = (MacroAction)i;
                foreach (MacroAction disabledAction in disabledActions)
                {
                    if (MacroActionMatchesDisableRule(macro, disabledAction))
                    {
                        ActionManager.DisableActionValue(macroMask, ActionManager.MacroOffset, i);
                        break;
                    }
                }
            }
        }

        private bool MacroActionMatchesRule(MacroAction macro, MacroAction rule)
        {
            if (macro == rule)
                return true;

            MacroAction ruleMovement = ActionManager.GetMovementMacro(rule);
            bool ruleIsPureMovement = rule == ruleMovement &&
                !ActionManager.IsCloseSkillAction(rule) &&
                !ActionManager.IsBindHealAction(rule);
            if (ruleIsPureMovement && ActionManager.GetMovementMacro(macro) == ruleMovement)
                return true;

            if (ActionManager.IsHorizontalSlashAction(rule) && ActionManager.IsHorizontalSlashAction(macro))
                return true;
            if (ActionManager.IsAntiAirSlashAction(rule) && ActionManager.IsAntiAirSlashAction(macro))
                return true;
            if (ActionManager.IsRangedToolAction(rule) && ActionManager.IsRangedToolAction(macro))
                return true;
            if (ActionManager.IsCloseSkillAction(rule) && ActionManager.IsCloseSkillAction(macro))
                return true;
            if (ActionManager.IsBindHealAction(rule) && ActionManager.IsBindHealAction(macro))
                return true;

            return false;
        }

        private bool MacroActionMatchesDisableRule(MacroAction macro, MacroAction rule)
        {
            if (macro == rule)
                return true;

            if (ActionManager.IsHorizontalSlashAction(rule) && ActionManager.IsHorizontalSlashAction(macro))
                return true;
            if (ActionManager.IsAntiAirSlashAction(rule) && ActionManager.IsAntiAirSlashAction(macro))
                return true;
            if (ActionManager.IsRangedToolAction(rule) && ActionManager.IsRangedToolAction(macro))
                return true;
            if (ActionManager.IsCloseSkillAction(rule) && ActionManager.IsCloseSkillAction(macro))
                return true;
            if (ActionManager.IsBindHealAction(rule) && ActionManager.IsBindHealAction(macro))
                return true;

            MacroAction ruleMovement = ActionManager.GetMovementMacro(rule);
            bool ruleIsPureMovement = rule == ruleMovement &&
                !ActionManager.IsCloseSkillAction(rule) &&
                !ActionManager.IsBindHealAction(rule);
            return ruleIsPureMovement && ActionManager.GetMovementMacro(macro) == ruleMovement;
        }

        private bool IsMacroEnabled(int[] macroMask, MacroAction macro)
        {
            return ActionManager.IsActionValueEnabled(macroMask, ActionManager.MacroOffset, (int)macro);
        }

        private bool IsMacroEnabledInActionMask(
            int[] actionMask,
            MacroAction macro,
            HeroController hero,
            HealthManager boss)
        {
            if (actionMask == null)
                return true;

            if (actionMask.Length == ActionManager.MacroActionCount)
                return IsMacroEnabled(actionMask, macro);

            int count = Mathf.Min(actionMask.Length, ActionManager.GetActionCount());
            for (int i = 0; i < count; i++)
            {
                if (actionMask[i] == 0)
                    continue;

                Action action = ActionManager.ActionIndexToAction(i, hero, boss, this);
                if (MacroActionMatchesRule(action.macro, macro))
                    return true;
            }

            return false;
        }

        private bool IsFacingBoss(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return true;

            bool bossIsRight = boss.transform.position.x >= hero.transform.position.x;
            return bossIsRight ? hero.cState.facingRight : !hero.cState.facingRight;
        }

        private bool EstimateBossFacingRight(HealthManager boss, Vector2 bossVelocity, Vector2 bossRelativeToHero)
        {
            if (bossVelocity.x > 0.05f)
                return true;
            if (bossVelocity.x < -0.05f)
                return false;

            // Most boss idle/antic states face the hero. Use that as a stable fallback
            // when velocity is near zero and no generic facing flag is exposed.
            return bossRelativeToHero.x < 0f;
        }

        private bool IsFarFromPreferredCenter(HeroController hero)
        {
            if (hero == null)
                return false;

            return Mathf.Abs(GetPreferredCenterX() - hero.transform.position.x) > 3f;
        }

        public int GetObservationSize()
        {
            return SingleFrameObservationSize * FrameStackSize;
        }

        public virtual void ResetObservationHistory()
        {
            stackedObservationHistory = null;
            stackedSingleFrameSize = 0;
            stackedBoss = null;
            stackedSceneName = null;
            pendingAttackCreditSeconds = 0f;
            pendingDodgeOverBossDamageWindowSeconds = 0f;
            lastRewardTime = -1f;
        }

        public virtual float CalculateReward(float[] previousObs, float[] currentObs, Action previousAction, int whoDied)
        {
            lastRewardComponents = new RewardComponents();

            if (previousObs == null || currentObs == null ||
                previousObs.Length < GetObservationSize() || currentObs.Length < GetObservationSize())
            {
                return 0f;
            }

            if (whoDied == 0)
                return -4000f;
            if (whoDied == 1)
                return 6000f;

            previousAction = previousAction ?? new Action();

            float reward = 0f;
            float elapsedRewardSeconds = GetElapsedRewardSeconds();
            float stepRewardScale = GetStepRewardScale(elapsedRewardSeconds);
            float prevHeroHP = previousObs[HeroHpIndex] * MaxHeroHP;
            float currHeroHP = currentObs[HeroHpIndex] * MaxHeroHP;
            float prevBossHP = previousObs[BossHpIndex] * MaxBossHP;
            float currBossHP = currentObs[BossHpIndex] * MaxBossHP;
            float phaseRewardMultiplier = GetPhaseRewardMultiplier(currentObs);
            float phaseReward = CalculatePhaseProgressReward(previousObs, currentObs);

            float bossHPLoss = Mathf.Max(0f, prevBossHP - currBossHP);
            float heroHPLoss = Mathf.Max(0f, prevHeroHP - currHeroHP);
            float heroHPGain = Mathf.Max(0f, currHeroHP - prevHeroHP);
            float prevSilk = previousObs[SilkIndex] * MaxObservedSilk;

            float damageReward = bossHPLoss * 32.5f * phaseRewardMultiplier;
            float heroDamagePenalty = heroHPLoss * 1000.0f;
            if (heroDamagePenalty > 0f)
            {
                heroDamagePenalty *= CalculateHeroDamagePenaltyMultiplier(currHeroHP);
            }
            float healReward = 0f;

            reward += damageReward;
            reward += phaseReward;
            reward -= heroDamagePenalty;
            lastRewardComponents.damageReward = damageReward;
            lastRewardComponents.phaseReward = phaseReward;
            lastRewardComponents.heroDamagePenalty = -heroDamagePenalty;

            int healedAmount = Mathf.RoundToInt(heroHPGain);
            if (healedAmount == 1)
            {
                healReward = 300.0f;
            }
            else if (healedAmount == 2)
            {
                healReward = 700f;
            }
            else if (healedAmount == 3)
            {
                healReward = 1100f;
            }
            else if (healedAmount == 4)
            {
                healReward = 1600f;
            }

            if (healReward > 0f)
            {
                healReward *= CalculateHealMultiplier(prevHeroHP);
            }

            if (IsValidBindAttempt(previousAction, prevHeroHP, prevSilk))
            {
                healReward += BindAttemptReward;
            }

            reward += healReward;
            lastRewardComponents.healReward = healReward;

            //reward += heroHPGain * 50.0f;

            float silkGain = (currentObs[SilkIndex] - previousObs[SilkIndex]) * MaxObservedSilk;
            float silkLoss = Mathf.Max(0f, -silkGain);
            float currSilk = currentObs[SilkIndex] * MaxObservedSilk;
            float silkHoldScale = currSilk >= 9f ? CalculateHighSilkHoldMultiplier(currHeroHP) : 1f;
            float resourceReward = ScaleStepReward(
                currSilk * (currSilk >= 9f ? 0.08f * silkHoldScale : 0.05f),
                stepRewardScale);
            reward += resourceReward;
            lastRewardComponents.resourceReward = resourceReward;

            float positionReward = 0f;
            float prevDistance = previousObs[DistanceIndex] * ArenaDiagonal;
            float currDistance = currentObs[DistanceIndex] * ArenaDiagonal;
            if (heroHPLoss <= 0f)
            {
                positionReward += (prevDistance - currDistance) * 0.5f;
            }

            if (currDistance > FarDistancePenaltyThreshold)
            {
                positionReward -= ScaleStepReward(0.2f, stepRewardScale);
            }

            float currHeroY = DenormalizePositionY(currentObs[1]);
            if (LowYWarningThreshold.HasValue && currHeroY < LowYWarningThreshold.Value)
            {
                positionReward -= ScaleStepReward(2f, stepRewardScale);
            }

            if (LowYDamagePenaltyThreshold.HasValue &&
                currHeroY < LowYDamagePenaltyThreshold.Value &&
                heroHPLoss > 0f)
            {
                positionReward -= heroHPLoss * 2000f;
            }

            positionReward += CalculateSafePlatformReward(previousObs, currentObs, previousAction, heroHPLoss, stepRewardScale);
            float collisionDangerReward = CalculateCollisionDangerReward(previousObs, currentObs, heroHPLoss, stepRewardScale);
            positionReward += collisionDangerReward;
            positionReward += CalculateBossDangerReward(previousObs, currentObs, previousAction, heroHPLoss, stepRewardScale);
            positionReward += CalculateDodgeOverBossOutcomeReward(previousAction, heroHPLoss, elapsedRewardSeconds);
            reward += positionReward;
            lastRewardComponents.positionReward = positionReward;

            float attackReward = CalculateAttackReward(previousObs, currentObs, previousAction, bossHPLoss, silkGain, elapsedRewardSeconds) * phaseRewardMultiplier;
            reward += attackReward;
            lastRewardComponents.attackReward = attackReward;

            if (previousAction.needle)
            {
                reward -= ScaleStepReward(20f, stepRewardScale);
            }

            if (previousAction.bind)
            {
                reward -= ScaleStepReward(15f, stepRewardScale);
            }

            if (previousAction.tool != ToolAction.None)
            {
                float toolAmountLoss = GetToolAmountLoss(previousObs, currentObs);
                if (toolAmountLoss > 0.001f)
                {
                    reward -= ToolAmmoUsePenalty;
                }
                else if (silkLoss > 0.1f)
                {
                    reward -= silkLoss*30f;
                    reward -=200;
                }
                else
                {
                    reward -= ScaleStepReward(5f, stepRewardScale);
                }
            }

            float baseSurvivalReward = ScaleStepReward(0.2f, stepRewardScale);
            float dodgeReward = CalculatePredictiveDodgeSuccessReward(previousObs, currentObs, heroHPLoss) * 1f * phaseRewardMultiplier;
            reward += baseSurvivalReward + dodgeReward;
            lastRewardComponents.baseSurvivalReward = baseSurvivalReward;
            lastRewardComponents.dodgeReward = dodgeReward;
            return reward;
        }

        protected virtual int GetPhaseFromObservation(float[] observations)
        {
            return 1;
        }

        private float GetPhaseRewardMultiplier(float[] observations)
        {
            int phase = Mathf.Clamp(GetPhaseFromObservation(observations), 1, 3);
            int exponent = Mathf.Max(0, phase - 1);
            return Mathf.Pow(PhaseRewardMultiplierBase, exponent);
        }

        private float CalculatePhaseProgressReward(float[] previousObs, float[] currentObs)
        {
            int previousPhase = Mathf.Clamp(GetPhaseFromObservation(previousObs), 1, 3);
            int currentPhase = Mathf.Clamp(GetPhaseFromObservation(currentObs), 1, 3);
            if (currentPhase <= previousPhase)
                return 0f;

            return (currentPhase - previousPhase) * PhaseProgressReward;
        }

        public RewardComponents GetLastRewardComponents()
        {
            return lastRewardComponents != null ? lastRewardComponents.Clone() : new RewardComponents();
        }

        private float CalculateHealMultiplier(float previousHeroHp)
        {
            float hpPercent = Mathf.Clamp01(previousHeroHp / Mathf.Max(1f, MaxHeroHP));
            float lowPercent = Mathf.Clamp01(HealMultiplierLowHpPercent);
            float highPercent = Mathf.Clamp01(HealMultiplierHighHpPercent);
            if (highPercent <= lowPercent)
            {
                return LowHealthHealMultiplier;
            }

            float t = Mathf.Clamp01((hpPercent - lowPercent) / (highPercent - lowPercent));
            return Mathf.Lerp(LowHealthHealMultiplier, HighHealthHealMultiplier, t);
        }

        private float CalculateHeroDamagePenaltyMultiplier(float currentHeroHp)
        {
            float hpPercent = Mathf.Clamp01(currentHeroHp / Mathf.Max(1f, MaxHeroHP));
            float lowPercent = Mathf.Clamp01(DamagePenaltyLowHpPercent);
            float highPercent = Mathf.Clamp01(DamagePenaltyHighHpPercent);
            if (highPercent <= lowPercent)
            {
                return LowHealthDamagePenaltyMultiplier;
            }

            float t = Mathf.Clamp01((hpPercent - lowPercent) / (highPercent - lowPercent));
            return Mathf.Lerp(LowHealthDamagePenaltyMultiplier, HighHealthDamagePenaltyMultiplier, t);
        }

        public virtual float[] CreateTerminalObservation(float[] previousObservations, int whoDied)
        {
            if (previousObservations == null)
                return null;

            float[] terminal = new float[previousObservations.Length];
            Array.Copy(previousObservations, terminal, previousObservations.Length);

            if (whoDied == 0 && terminal.Length > HeroHpIndex)
            {
                terminal[HeroHpIndex] = 0f;
            }
            else if (whoDied == 1 && terminal.Length > BossHpIndex)
            {
                terminal[BossHpIndex] = 0f;
            }

            return terminal;
        }

        public virtual bool IsHeroStuck(HeroController hero)
        {
            if (hero == null)
                return false;

            return hero.transform.position.y < MinPosY - 5f;
        }

        public ScreenCapture GetScreenCapture()
        {
            return null;
        }

        public float GetMaxHP()
        {
            return MaxBossHP;
        }

        protected virtual void WriteBossSpecificObservations(float[] observations, int offset, HeroController hero, HealthManager boss)
        {
        }

        protected virtual bool IsBossAttackOpportunityState()
        {
            return false;
        }

        protected virtual bool IsBossSafeForBindState()
        {
            return IsBossAttackOpportunityState();
        }

        protected virtual bool IsBossDangerousState()
        {
            return false;
        }

        protected virtual bool IsBossGroundRushState()
        {
            return false;
        }

        private bool IsValidBindAttempt(Action previousAction, float previousHeroHp, float previousSilk)
        {
            return previousAction != null &&
                previousAction.optionStarted &&
                previousAction.bind &&
                previousHeroHp > 0f &&
                previousHeroHp < MaxHeroHP &&
                previousSilk >= 9f;
        }

        private float CalculateHighSilkHoldMultiplier(float currentHeroHp)
        {
            float hpPercent = Mathf.Clamp01(currentHeroHp / Mathf.Max(1f, MaxHeroHP));
            float lowPercent = Mathf.Clamp01(HighSilkHoldLowHpPercent);
            float highPercent = Mathf.Clamp01(HighSilkHoldHighHpPercent);
            if (highPercent <= lowPercent)
            {
                return HighSilkHoldLowHpMultiplier;
            }

            float t = Mathf.Clamp01((hpPercent - lowPercent) / (highPercent - lowPercent));
            return Mathf.Lerp(HighSilkHoldLowHpMultiplier, HighSilkHoldHighHpMultiplier, t);
        }

        private float CalculateAttackReward(
            float[] previousObs,
            float[] currentObs,
            Action previousAction,
            float bossHPLoss,
            float silkGain,
            float elapsedSeconds)
        {
            float reward = 0f;
            bool attackOpportunity = IsAttackOpportunity(previousObs);

            if (attackOpportunity)
            {
                if (!previousAction.attack && MissedAttackOpportunityPenalty > 0f)
                {
                    reward -= MissedAttackOpportunityPenalty;
                }

                if (previousAction.tool != ToolAction.None && AttackOpportunityToolPenalty > 0f)
                {
                    reward -= AttackOpportunityToolPenalty;
                }
            }

            if (previousAction.attack)
            {
                reward += ScaleStepReward(AttackPressReward, GetStepRewardScale(elapsedSeconds));

                if (pendingAttackCreditSeconds <= 0f)
                {
                    pendingAttackCreditSeconds = AttackCreditWindowSeconds;
                }

                if (IsDirectedAttackOpportunity(previousObs, previousAction))
                {
                    reward += AttackOpportunityReward;
                }
            }

            if (pendingAttackCreditSeconds <= 0f)
                return reward;

            if (bossHPLoss > 0f)
            {
                reward += AttackCreditHitReward;
                if (silkGain > 0f)
                {
                    reward += silkGain * AttackSilkGainReward;
                }

                pendingAttackCreditSeconds = 0f;
                return reward;
            }

            pendingAttackCreditSeconds -= Mathf.Max(Time.fixedDeltaTime, elapsedSeconds);
            if (pendingAttackCreditSeconds <= 0f)
            {
                reward -= EmptyAttackPenalty;
            }

            return reward;
        }

        private float CalculateBossDangerReward(
            float[] previousObs,
            float[] currentObs,
            Action previousAction,
            float heroHPLoss,
            float stepRewardScale)
        {
            if (!IsBossDangerousState() ||
                BossDangerDistance <= 0f ||
                (BossDangerGreedyActionPenalty <= 0f &&
                    BossDangerGreedyDamagePenalty <= 0f &&
                    BossDangerDodgeReward <= 0f &&
                    BossDangerRetreatRewardScale <= 0f))
            {
                return 0f;
            }

            if (!TryGetAttackGeometry(
                    previousObs,
                    out float relX,
                    out float relY,
                    out float absX,
                    out float absY,
                    out bool _))
            {
                return 0f;
            }

            if (absX > BossDangerDistance || absY > BossDangerVerticalTolerance)
                return 0f;

            previousAction = previousAction ?? new Action();
            float horizontalDepth = Mathf.Clamp01((BossDangerDistance - absX) / BossDangerDistance);
            float verticalDepth = BossDangerVerticalTolerance > 0f
                ? Mathf.Clamp01((BossDangerVerticalTolerance - absY) / BossDangerVerticalTolerance)
                : 1f;
            float dangerDepth = Mathf.Clamp01(Mathf.Max(horizontalDepth, verticalDepth));

            bool greedyAction = IsGreedyAction(previousAction);
            bool defensiveAction = IsDefensiveDangerAction(previousObs, currentObs, previousAction, relX);
            float reward = 0f;

            if (greedyAction)
            {
                reward -= ScaleStepReward(BossDangerGreedyActionPenalty * dangerDepth, stepRewardScale);
                if (heroHPLoss > 0f)
                {
                    reward -= heroHPLoss * BossDangerGreedyDamagePenalty * Mathf.Max(0.5f, dangerDepth);
                }
            }

            float prevDistance = previousObs[DistanceIndex] * ArenaDiagonal;
            float currDistance = currentObs[DistanceIndex] * ArenaDiagonal;
            if (heroHPLoss <= 0f)
            {
                if (defensiveAction)
                {
                    reward += ScaleStepReward(BossDangerDodgeReward * dangerDepth, stepRewardScale);
                }

                if (currDistance > prevDistance && BossDangerRetreatRewardScale > 0f)
                {
                    reward += (currDistance - prevDistance) * BossDangerRetreatRewardScale * Mathf.Max(0.5f, dangerDepth);
                }
            }

            return reward;
        }

        private float CalculateDodgeOverBossOutcomeReward(Action previousAction, float heroHPLoss, float elapsedSeconds)
        {
            previousAction = previousAction ?? new Action();
            if (ActionManager.GetMovementMacro(previousAction.macro) == MacroAction.DodgeOverBossHighJump)
            {
                pendingDodgeOverBossDamageWindowSeconds = Mathf.Max(
                    pendingDodgeOverBossDamageWindowSeconds,
                    DodgeOverBossDamageWindowSeconds);
            }

            if (pendingDodgeOverBossDamageWindowSeconds <= 0f)
                return 0f;

            if (heroHPLoss > 0f)
            {
                pendingDodgeOverBossDamageWindowSeconds = 0f;
                return -heroHPLoss * DodgeOverBossDamagePenalty;
            }

            pendingDodgeOverBossDamageWindowSeconds -= Mathf.Max(Time.fixedDeltaTime, elapsedSeconds);
            return 0f;
        }

        private float CalculatePredictiveDodgeSuccessReward(
            float[] previousObs,
            float[] currentObs,
            float heroHPLoss)
        {
            if (PredictiveDodgeSuccessReward <= 0f || heroHPLoss > 0f)
                return 0f;

            if (IsCurrentHeroProtected(currentObs))
                return 0f;

            if (HasEscapedBossVelocityDanger(previousObs, currentObs))
                return PredictiveDodgeSuccessReward;

            bool currentWouldHaveHit = WouldStationaryHeroHaveBeenHit(currentObs);
            if (!currentWouldHaveHit)
                return 0f;

            bool previousWouldHaveHit = WouldStationaryHeroHaveBeenHit(previousObs);
            return previousWouldHaveHit ? 0f : PredictiveDodgeSuccessReward;
        }

        private bool HasEscapedBossVelocityDanger(float[] previousObs, float[] currentObs)
        {
            if (!IsBossDangerousState())
                return false;

            if (!TryReadStackedFrame(
                    previousObs,
                    0,
                    out Vector2 previousHeroPos,
                    out Vector2 previousBossPos,
                    out Vector2 previousHeroVel,
                    out Vector2 previousBossVel))
            {
                return false;
            }

            if (!TryReadStackedFrame(
                    currentObs,
                    0,
                    out Vector2 currentHeroPos,
                    out Vector2 currentBossPos,
                    out Vector2 currentHeroVel,
                    out Vector2 currentBossVel))
            {
                return false;
            }

            Vector2 bossVelocity = currentBossVel.sqrMagnitude >= previousBossVel.sqrMagnitude
                ? currentBossVel
                : previousBossVel;
            if (bossVelocity.magnitude < PredictiveDodgeMinBossMotion)
                return false;

            Vector2 attackDirection = bossVelocity.normalized;
            if (!IsInBossVelocityDangerPath(previousHeroPos, previousBossPos, attackDirection))
                return false;

            if (IsInBossVelocityDangerPath(currentHeroPos, currentBossPos, attackDirection))
                return false;

            float heroMotion = Vector2.Distance(previousHeroPos, currentHeroPos) +
                Mathf.Max(previousHeroVel.magnitude, currentHeroVel.magnitude) * PredictiveDodgeStepSeconds;
            return heroMotion >= PredictiveDodgeMinHeroMotion;
        }

        private bool IsInBossVelocityDangerPath(Vector2 heroPos, Vector2 bossPos, Vector2 attackDirection)
        {
            if (attackDirection.sqrMagnitude <= 0.0001f)
                return false;

            Vector2 direction = attackDirection.normalized;
            Vector2 rel = heroPos - bossPos;
            float forward = Vector2.Dot(rel, direction);
            if (forward < 0f || forward > PredictiveDodgeVelocityEscapeDistance)
                return false;

            Vector2 closest = bossPos + direction * forward;
            float lateralDistance = Vector2.Distance(heroPos, closest);
            float lateralRadius = Mathf.Max(
                PredictiveDodgeHitRadius + PredictiveDodgeVelocityEscapeExitMargin,
                BossDangerVerticalTolerance * 0.5f);
            return lateralDistance <= lateralRadius;
        }

        private bool WouldStationaryHeroHaveBeenHit(float[] observations)
        {
            if (observations == null || observations.Length < GetObservationSize())
                return false;

            int newestFrame = Mathf.Clamp(PredictiveDodgeNewestHistoryFrame, 1, FrameStackSize - 1);
            int oldestFrame = Mathf.Clamp(PredictiveDodgeOldestHistoryFrame, newestFrame, FrameStackSize - 1);
            float stepSeconds = Mathf.Max(0.001f, PredictiveDodgeStepSeconds);

            if (!TryReadStackedFrame(
                    observations,
                    0,
                    out Vector2 currentHeroPos,
                    out Vector2 currentBossPos,
                    out Vector2 currentHeroVel,
                    out Vector2 currentBossVel))
            {
                return false;
            }

            for (int frameIndex = newestFrame; frameIndex <= oldestFrame; frameIndex++)
            {
                if (!TryReadStackedFrame(
                        observations,
                        frameIndex,
                        out Vector2 historicalHeroPos,
                        out Vector2 bossPos,
                        out Vector2 historicalHeroVel,
                        out Vector2 bossVel))
                {
                    continue;
                }

                if (!TryReadStackedFrame(
                        observations,
                        frameIndex - 1,
                        out Vector2 newerHeroPos,
                        out Vector2 newerBossPos,
                        out Vector2 newerHeroVel,
                        out Vector2 newerBossVel))
                {
                    continue;
                }

                float frameDeltaSeconds = Mathf.Max(
                    stepSeconds,
                    (GetFrameAgeSteps(frameIndex) - GetFrameAgeSteps(frameIndex - 1)) * stepSeconds);
                Vector2 heroAcceleration = ClampMagnitude((newerHeroVel - historicalHeroVel) / frameDeltaSeconds, PredictiveDodgeMaxAcceleration);
                Vector2 bossAcceleration = ClampMagnitude((newerBossVel - bossVel) / frameDeltaSeconds, PredictiveDodgeMaxAcceleration);
                float lookaheadSeconds = GetFrameAgeSteps(frameIndex) * stepSeconds;
                Vector2 projectedBossMotion = bossVel * lookaheadSeconds + 0.5f * bossAcceleration * lookaheadSeconds * lookaheadSeconds;
                Vector2 projectedBossPos = bossPos + projectedBossMotion;
                Vector2 attackDirection = projectedBossMotion.sqrMagnitude > 0.0001f ? projectedBossMotion : bossVel * lookaheadSeconds;

                if (attackDirection.magnitude < PredictiveDodgeMinBossMotion * lookaheadSeconds)
                    continue;

                Vector2 bossToStationaryHero = historicalHeroPos - bossPos;
                if (bossToStationaryHero.sqrMagnitude > 0.0001f &&
                    Vector2.Dot(attackDirection.normalized, bossToStationaryHero.normalized) < 0.25f)
                {
                    continue;
                }

                float heroMotionSignal =
                    Vector2.Distance(currentHeroPos, historicalHeroPos) +
                    historicalHeroVel.magnitude * lookaheadSeconds +
                    0.5f * heroAcceleration.magnitude * lookaheadSeconds * lookaheadSeconds;
                if (heroMotionSignal < PredictiveDodgeMinHeroMotion)
                    continue;

                Vector2 attackFront = projectedBossPos + attackDirection.normalized * PredictiveDodgeAttackLeadDistance;
                float stationaryDistance = DistancePointToSegment(historicalHeroPos, projectedBossPos, attackFront);
                if (stationaryDistance > PredictiveDodgeHitRadius)
                    continue;

                float currentDistance = DistancePointToSegment(currentHeroPos, projectedBossPos, attackFront);
                if (currentDistance > PredictiveDodgeHitRadius + PredictiveDodgeSafeExitMargin)
                    return true;
            }

            return false;
        }

        private static int GetFrameAgeSteps(int frameIndex)
        {
            if (frameIndex < 0)
                return 0;
            if (frameIndex >= FrameSampleOffsets.Length)
                return FrameSampleOffsets[FrameSampleOffsets.Length - 1];

            return FrameSampleOffsets[frameIndex];
        }

        private bool IsCurrentHeroProtected(float[] observations)
        {
            if (observations == null || observations.Length <= 20)
                return false;

            bool heroRecoiling = observations[19] > 0.5f;
            bool heroInvulnerable = observations[20] > 0.5f;
            return heroRecoiling || heroInvulnerable;
        }

        private bool TryReadStackedFrame(
            float[] observations,
            int frameIndex,
            out Vector2 heroPos,
            out Vector2 bossPos,
            out Vector2 heroVel,
            out Vector2 bossVel)
        {
            heroPos = Vector2.zero;
            bossPos = Vector2.zero;
            heroVel = Vector2.zero;
            bossVel = Vector2.zero;

            if (observations == null || frameIndex < 0 || frameIndex >= FrameStackSize)
                return false;

            int offset = frameIndex * SingleFrameObservationSize;
            if (offset + 8 >= observations.Length)
                return false;

            heroPos = new Vector2(
                DenormalizePositionX(observations[offset]),
                DenormalizePositionY(observations[offset + 1]));
            heroVel = new Vector2(
                DenormalizeSigned(observations[offset + 2], MaxHeroVelocity),
                DenormalizeSigned(observations[offset + 3], MaxHeroVelocity));
            bossPos = new Vector2(
                DenormalizePositionX(observations[offset + 5]),
                DenormalizePositionY(observations[offset + 6]));
            bossVel = new Vector2(
                DenormalizeSigned(observations[offset + 7], MaxBossVelocity),
                DenormalizeSigned(observations[offset + 8], MaxBossVelocity));
            return true;
        }

        private static Vector2 ClampMagnitude(Vector2 value, float maxMagnitude)
        {
            if (maxMagnitude <= 0f || value.sqrMagnitude <= maxMagnitude * maxMagnitude)
                return value;

            return value.normalized * maxMagnitude;
        }

        private static float DistancePointToSegment(Vector2 point, Vector2 segmentStart, Vector2 segmentEnd)
        {
            Vector2 segment = segmentEnd - segmentStart;
            float lengthSquared = segment.sqrMagnitude;
            if (lengthSquared <= 0.000001f)
                return Vector2.Distance(point, segmentStart);

            float t = Mathf.Clamp01(Vector2.Dot(point - segmentStart, segment) / lengthSquared);
            Vector2 closest = segmentStart + segment * t;
            return Vector2.Distance(point, closest);
        }

        private bool IsGreedyAction(Action action)
        {
            return action != null &&
                (action.attack ||
                    action.needle ||
                    action.bind ||
                    action.tool != ToolAction.None);
        }

        private bool IsDefensiveDangerAction(
            float[] previousObs,
            float[] currentObs,
            Action action,
            float relX)
        {
            if (action == null)
                return false;

            MacroAction movementMacro = ActionManager.GetMovementMacro(action.macro);
            if (movementMacro == MacroAction.DodgeOverBossHighJump)
                return false;

            if (movementMacro == MacroAction.RetreatFromBossDodge ||
                movementMacro == MacroAction.MoveToCenterSafely ||
                movementMacro == MacroAction.CrossUnderAirBoss ||
                action.jump)
            {
                return true;
            }

            MoveDirection away = relX >= 0f ? MoveDirection.Left : MoveDirection.Right;
            if (action.move == away)
                return true;

            float prevDistance = previousObs[DistanceIndex] * ArenaDiagonal;
            float currDistance = currentObs[DistanceIndex] * ArenaDiagonal;
            return action.dash && currDistance > prevDistance;
        }

        private float CalculateCollisionDangerReward(
            float[] previousObs,
            float[] currentObs,
            float heroHPLoss,
            float stepRewardScale)
        {
            if (CollisionDangerDistance <= 0f ||
                (CollisionDangerPenalty <= 0f &&
                    CollisionDamagePenalty <= 0f &&
                    CollisionDangerRetreatRewardScale <= 0f))
            {
                return 0f;
            }

            bool protectedState =
                currentObs[15] > 0.5f ||
                currentObs[19] > 0.5f ||
                currentObs[20] > 0.5f;
            if (protectedState)
            {
                return 0f;
            }

            float reward = 0f;
            float prevDistance = previousObs[DistanceIndex] * ArenaDiagonal;
            float currDistance = currentObs[DistanceIndex] * ArenaDiagonal;

            if (currDistance < CollisionDangerDistance && CollisionDangerPenalty > 0f)
            {
                float dangerDepth = Mathf.Clamp01((CollisionDangerDistance - currDistance) / CollisionDangerDistance);
                reward -= ScaleStepReward(dangerDepth * CollisionDangerPenalty, stepRewardScale);
            }

            if (heroHPLoss > 0f && currDistance < CollisionDangerDistance && CollisionDamagePenalty > 0f)
            {
                float dangerDepth = Mathf.Clamp01((CollisionDangerDistance - currDistance) / CollisionDangerDistance);
                reward -= heroHPLoss * CollisionDamagePenalty * Mathf.Max(0.5f, dangerDepth);
            }

            if (prevDistance < CollisionDangerDistance &&
                currDistance > prevDistance &&
                CollisionDangerRetreatRewardScale > 0f)
            {
                float previousClamped = Mathf.Clamp(prevDistance, 0f, CollisionDangerDistance);
                float currentClamped = Mathf.Clamp(currDistance, 0f, CollisionDangerDistance);
                reward += Mathf.Max(0f, currentClamped - previousClamped) * CollisionDangerRetreatRewardScale;
            }

            return reward;
        }

        private float GetElapsedRewardSeconds()
        {
            float now = Time.fixedTime;
            if (lastRewardTime < 0f)
            {
                lastRewardTime = now;
                return Time.fixedDeltaTime;
            }

            float elapsed = Mathf.Max(Time.fixedDeltaTime, now - lastRewardTime);
            lastRewardTime = now;
            return elapsed;
        }

        private float GetStepRewardScale(float elapsedSeconds)
        {
            return Mathf.Max(0.001f, elapsedSeconds) / RewardReferenceStepSeconds;
        }

        private float ScaleStepReward(float value, float stepRewardScale)
        {
            return value * stepRewardScale;
        }

        private bool IsFlatAttackOpportunity(float[] observations)
        {
            return TryGetAttackGeometry(
                    observations,
                    out float relX,
                    out float relY,
                    out float absX,
                    out float absY,
                    out bool facingBoss) &&
                facingBoss &&
                absX >= AttackRangeMin &&
                absX <= AttackRangeMax &&
                absY <= AttackVerticalTolerance &&
                IsBossAttackOpportunityState();
        }

        private bool IsAttackOpportunity(float[] observations)
        {
            return IsFlatAttackOpportunity(observations) ||
                IsVerticalAttackOpportunity(observations);
        }

        private bool IsVerticalAttackOpportunity(float[] observations)
        {
            return TryGetAttackGeometry(
                    observations,
                    out float relX,
                    out float relY,
                    out float absX,
                    out float absY,
                    out bool facingBoss) &&
                IsVerticalAttackOpportunityGeometry(relY, absX, absY) &&
                IsBossAttackOpportunityState();
        }

        private bool IsVerticalAttackOpportunityGeometry(float relY, float absX, float absY)
        {
            return relY >= AntiAirPriorityMinRelY &&
                relY <= VerticalAttackRangeMax &&
                absY >= VerticalAttackRangeMin &&
                absY <= VerticalAttackRangeMax &&
                absX <= VerticalAttackHorizontalTolerance;
        }

        private bool IsDirectedAttackOpportunity(float[] observations, Action action)
        {
            if (!TryGetAttackGeometry(
                    observations,
                    out float relX,
                    out float relY,
                    out float absX,
                    out float absY,
                    out bool facingBoss))
            {
                return false;
            }

            if (!IsBossAttackOpportunityState())
                return false;

            LookDirection look = action != null ? action.look : LookDirection.None;
            if (look == LookDirection.Up)
            {
                return IsVerticalAttackOpportunityGeometry(relY, absX, absY);
            }

            if (look == LookDirection.Down)
            {
                float downwardDistance = -relY;
                return downwardDistance >= VerticalAttackRangeMin &&
                    downwardDistance <= VerticalAttackRangeMax &&
                    absX <= VerticalAttackHorizontalTolerance;
            }

            return facingBoss &&
                absX >= AttackRangeMin &&
                absX <= AttackRangeMax &&
                absY <= AttackVerticalTolerance;
        }

        private bool TryGetAttackGeometry(
            float[] observations,
            out float relX,
            out float relY,
            out float absX,
            out float absY,
            out bool facingBoss)
        {
            relX = 0f;
            relY = 0f;
            absX = 0f;
            absY = 0f;
            facingBoss = false;

            if (observations == null || observations.Length <= HeroFacingIndex)
                return false;

            float heroX = DenormalizePositionX(observations[0]);
            float heroY = DenormalizePositionY(observations[1]);
            float bossX = DenormalizePositionX(observations[5]);
            float bossY = DenormalizePositionY(observations[6]);
            bool heroFacingRight = observations[HeroFacingIndex] > 0.5f;
            bool bossIsRight = bossX >= heroX;
            facingBoss = bossIsRight ? heroFacingRight : !heroFacingRight;
            relX = bossX - heroX;
            relY = bossY - heroY;
            absX = Mathf.Abs(relX);
            absY = Mathf.Abs(relY);
            return true;
        }

        private void ApplySafePlatformMask(int[] mask, HeroController hero)
        {
            if (!SafeMinPosX.HasValue || !SafeMaxPosX.HasValue || hero == null)
                return;

            float heroX = hero.transform.position.x;
            float safeMin = SafeMinPosX.Value;
            float safeMax = SafeMaxPosX.Value;
            float margin = Mathf.Max(BoundaryMaskMargin, SafePlatformMaskMargin);
            bool nearLeftEdge = heroX <= safeMin + margin;
            bool nearRightEdge = heroX >= safeMax - margin;

            if (nearLeftEdge)
            {
                ActionManager.DisableActionValue(mask, ActionManager.MoveOffset, (int)MoveDirection.Left);
            }
            if (nearRightEdge)
            {
                ActionManager.DisableActionValue(mask, ActionManager.MoveOffset, (int)MoveDirection.Right);
            }

            // Keep jump/dash available near an edge: outward movement is masked above,
            // and the execution layer clamps no-direction dash into the inward direction.
        }

        private float CalculateSafePlatformReward(
            float[] previousObs,
            float[] currentObs,
            Action previousAction,
            float heroHPLoss,
            float stepRewardScale)
        {
            if (!SafeMinPosX.HasValue || !SafeMaxPosX.HasValue)
                return 0f;

            float safeMin = SafeMinPosX.Value;
            float safeMax = SafeMaxPosX.Value;
            float warningMargin = Mathf.Max(0.1f, SafePlatformWarningMargin);
            float prevHeroX = DenormalizePositionX(previousObs[0]);
            float currHeroX = DenormalizePositionX(currentObs[0]);
            float currBossX = DenormalizePositionX(currentObs[5]);
            float reward = 0f;

            bool outsideLeft = currHeroX < safeMin;
            bool outsideRight = currHeroX > safeMax;
            if (outsideLeft || outsideRight)
            {
                reward -= ScaleStepReward(SafePlatformOutsidePenalty, stepRewardScale);
                if (heroHPLoss > 0f)
                {
                    reward -= heroHPLoss * SafePlatformDamagePenalty;
                }
            }

            if (currHeroX < safeMin + warningMargin)
            {
                float edgeDepth = Mathf.Clamp01((safeMin + warningMargin - currHeroX) / warningMargin);
                reward -= ScaleStepReward(edgeDepth * SafePlatformEdgePenalty, stepRewardScale);
                reward += (currHeroX - prevHeroX) * SafePlatformCenteringRewardScale;

                if (previousAction.move == MoveDirection.Left)
                {
                    reward -= ScaleStepReward(2f, stepRewardScale);
                }

                reward += CalculateEdgePressureReward(
                    isLeftEdge: true,
                    prevHeroX: prevHeroX,
                    currHeroX: currHeroX,
                    currBossX: currBossX,
                    previousAction: previousAction,
                    stepRewardScale: stepRewardScale);
            }
            else if (currHeroX > safeMax - warningMargin)
            {
                float edgeDepth = Mathf.Clamp01((currHeroX - (safeMax - warningMargin)) / warningMargin);
                reward -= ScaleStepReward(edgeDepth * SafePlatformEdgePenalty, stepRewardScale);
                reward += (prevHeroX - currHeroX) * SafePlatformCenteringRewardScale;

                if (previousAction.move == MoveDirection.Right)
                {
                    reward -= ScaleStepReward(2f, stepRewardScale);
                }

                reward += CalculateEdgePressureReward(
                    isLeftEdge: false,
                    prevHeroX: prevHeroX,
                    currHeroX: currHeroX,
                    currBossX: currBossX,
                    previousAction: previousAction,
                    stepRewardScale: stepRewardScale);
            }

            if ((outsideLeft || outsideRight) && previousAction.jump)
            {
                reward -= ScaleStepReward(5f, stepRewardScale);
            }

            if (previousAction.dash &&
                (currHeroX < safeMin + SafePlatformMaskMargin || currHeroX > safeMax - SafePlatformMaskMargin))
            {
                reward -= ScaleStepReward(8f, stepRewardScale);
            }

            return reward;
        }

        private float CalculateEdgePressureReward(
            bool isLeftEdge,
            float prevHeroX,
            float currHeroX,
            float currBossX,
            Action previousAction,
            float stepRewardScale)
        {
            if (Mathf.Abs(currBossX - currHeroX) > SafePlatformPressureBossDistance)
                return 0f;

            float inwardDelta = isLeftEdge ? currHeroX - prevHeroX : prevHeroX - currHeroX;
            MoveDirection inwardMove = isLeftEdge ? MoveDirection.Right : MoveDirection.Left;
            MoveDirection outwardMove = isLeftEdge ? MoveDirection.Left : MoveDirection.Right;
            float reward = inwardDelta * SafePlatformPressureCenteringRewardScale;

            if (previousAction.move == inwardMove)
            {
                reward += ScaleStepReward(SafePlatformPressureMoveInwardReward, stepRewardScale);
            }
            else if (previousAction.move == outwardMove)
            {
                reward -= ScaleStepReward(SafePlatformPressureMoveOutwardPenalty, stepRewardScale);
            }
            else
            {
                reward -= ScaleStepReward(SafePlatformPressureNoMovePenalty, stepRewardScale);
            }

            return reward;
        }

        private float[] BuildStackedObservation(float[] currentFrame, HealthManager boss)
        {
            int frameSize = currentFrame.Length;
            string sceneName = SceneManager.GetActiveScene().name;
            bool shouldReset =
                stackedObservationHistory == null ||
                stackedSingleFrameSize != frameSize ||
                stackedBoss != boss ||
                stackedSceneName != sceneName;

            if (shouldReset)
            {
                stackedSingleFrameSize = frameSize;
                stackedBoss = boss;
                stackedSceneName = sceneName;
                stackedObservationHistory = new float[frameSize * FrameHistorySize];

                for (int i = 0; i < FrameHistorySize; i++)
                {
                    Array.Copy(currentFrame, 0, stackedObservationHistory, i * frameSize, frameSize);
                }
            }
            else
            {
                for (int i = FrameHistorySize - 1; i >= 1; i--)
                {
                    Array.Copy(
                        stackedObservationHistory,
                        (i - 1) * frameSize,
                        stackedObservationHistory,
                        i * frameSize,
                        frameSize);
                }

                Array.Copy(currentFrame, 0, stackedObservationHistory, 0, frameSize);
            }

            float[] output = new float[frameSize * FrameStackSize];
            for (int i = 0; i < FrameStackSize; i++)
            {
                int sourceFrame = Mathf.Clamp(GetFrameAgeSteps(i), 0, FrameHistorySize - 1);
                Array.Copy(
                    stackedObservationHistory,
                    sourceFrame * frameSize,
                    output,
                    i * frameSize,
                    frameSize);
            }

            return output;
        }

        private void WriteCoreObservations(float[] observations, HeroController hero, HealthManager boss)
        {
            Vector2 heroPos = hero.transform.position;
            Rigidbody2D heroRb = hero.GetComponent<Rigidbody2D>();
            Vector2 heroVel = heroRb ? heroRb.linearVelocity : Vector2.zero;

            Vector2 bossPos = boss.transform.position;
            Rigidbody2D bossRb = boss.GetComponent<Rigidbody2D>();
            Vector2 bossVel = bossRb ? bossRb.linearVelocity : Vector2.zero;

            Vector2 relPos = bossPos - heroPos;
            Vector2 relVel = bossVel - heroVel;

            observations[0] = NormalizePositionX(heroPos.x);
            observations[1] = NormalizePositionY(heroPos.y);
            observations[2] = NormalizeSigned(heroVel.x, MaxHeroVelocity);
            observations[3] = NormalizeSigned(heroVel.y, MaxHeroVelocity);
            observations[4] = Mathf.Clamp01(hero.playerData.health / MaxHeroHP);

            observations[5] = NormalizePositionX(bossPos.x);
            observations[6] = NormalizePositionY(bossPos.y);
            observations[7] = NormalizeSigned(bossVel.x, MaxBossVelocity);
            observations[8] = NormalizeSigned(bossVel.y, MaxBossVelocity);
            observations[9] = Mathf.Clamp01(boss.hp / MaxBossHP);

            observations[10] = NormalizeSigned(relPos.x, ArenaWidth);
            observations[11] = NormalizeSigned(relPos.y, ArenaHeight);
            observations[12] = NormalizeSigned(relVel.x, MaxHeroVelocity + MaxBossVelocity);
            observations[13] = NormalizeSigned(relVel.y, MaxHeroVelocity + MaxBossVelocity);
            observations[14] = Mathf.Clamp01(Vector2.Distance(heroPos, bossPos) / ArenaDiagonal);

            observations[15] = hero.cState.dashing ? 1f : 0f;
            observations[16] = hero.cState.isSprinting ? 1f : 0f;
            observations[17] = hero.cState.attacking ? 1f : 0f;
            observations[18] = hero.cState.jumping ? 1f : 0f;
            observations[19] = hero.cState.recoiling ? 1f : 0f;
            observations[20] = hero.cState.Invulnerable ? 1f : 0f;
            observations[21] = hero.cState.facingRight ? 1f : 0f;
            observations[22] = IsFacingBoss(hero, boss) ? 1f : 0f;
            observations[23] = EstimateBossFacingRight(boss, bossVel, relPos) ? 1f : 0f;
        }

        private void WriteResourceObservations(float[] observations)
        {
            if (!PlayerData.HasInstance)
                return;

            PlayerData playerData = PlayerData.instance;
            observations[SilkIndex] = Mathf.Clamp01(playerData.silk / MaxObservedSilk);
            observations[SilkIndex + 1] = Mathf.Clamp01(playerData.CurrentSilkMax / MaxObservedSilk);
            int maxHealth = Mathf.Max(1, playerData.CurrentMaxHealth);
            bool injured = playerData.health < maxHealth;
            observations[SilkIndex + 2] = injured && playerData.silk >= 9 ? 1f : 0f;
            observations[SilkIndex + 3] = Mathf.Clamp01(Mathf.Max(0f, 9f - playerData.silk) / 9f);

            WriteToolSlotObservations(observations, AttackToolBinding.Neutral, ToolSlotStartIndex);
            WriteToolSlotObservations(observations, AttackToolBinding.Up, ToolSlotStartIndex + ToolSlotSize);
            WriteToolSlotObservations(observations, AttackToolBinding.Down, ToolSlotStartIndex + ToolSlotSize * 2);
        }

        private void WriteToolSlotObservations(float[] observations, AttackToolBinding binding, int offset)
        {
            try
            {
                ToolItem tool = ToolItemManager.GetBoundAttackTool(binding, ToolEquippedReadSource.Active);
                if (!tool)
                    return;

                observations[offset] = 1f;
                observations[offset + 1] = tool.Type == ToolItemType.Skill ? 1f : 0f;

                if (tool.Type == ToolItemType.Skill)
                {
                    int skillCost = Mathf.Max(1, PlayerData.instance.SilkSkillCost);
                    observations[offset + 2] = Mathf.Clamp01(PlayerData.instance.silk / (float)skillCost);
                    observations[offset + 3] = 1f;
                    observations[offset + 4] = Mathf.Clamp01(skillCost / MaxObservedSkillCost);
                    return;
                }

                ToolItemsData.Data savedData = tool.SavedData;
                int capacity = Mathf.Max(1, ToolItemManager.GetToolStorageAmount(tool));
                observations[offset + 2] = Mathf.Clamp01(savedData.AmountLeft / (float)capacity);
                observations[offset + 3] = Mathf.Clamp01(capacity / MaxObservedToolCapacity);
                observations[offset + 4] = Mathf.Clamp01(tool.Usage.SilkRequired / MaxObservedSkillCost);
            }
            catch (Exception e)
            {
                RLManager.StaticLogger?.LogWarning($"[BossEncounterBase] Failed to read tool slot {binding}: {e.Message}");
            }
        }

        private bool CanBind(HeroController hero)
        {
            if (hero == null || hero.playerData == null)
                return false;

            int maxHealth = Mathf.Max(1, hero.playerData.CurrentMaxHealth);
            if (hero.playerData.health >= maxHealth)
                return false;

            if (!PlayerData.HasInstance)
                return false;

            return PlayerData.instance.silk >= 9;
        }

        private bool ShouldReserveSilkForBind(HeroController hero)
        {
            if (hero == null || hero.playerData == null || !PlayerData.HasInstance)
                return false;

            int maxHealth = Mathf.Max(1, hero.playerData.CurrentMaxHealth);
            bool injured = hero.playerData.health < maxHealth;
            return injured && PlayerData.instance.silk < 9;
        }

        private bool ShouldProtectBindResources(HeroController hero, HealthManager boss)
        {
            if (hero == null || hero.playerData == null || !PlayerData.HasInstance)
                return false;

            int maxHealth = Mathf.Max(1, hero.playerData.CurrentMaxHealth);
            float hpPercent = hero.playerData.health / (float)maxHealth;
            if (hpPercent > LowHealthBindResourceMaskHpPercent)
                return false;

            if (PlayerData.instance.silk < 9)
                return false;

            if (IsBossSafeForBindState())
                return true;

            if (boss == null)
                return false;

            return Vector2.Distance(hero.transform.position, boss.transform.position) >= SafeBindFarDistanceThreshold;
        }

        private bool CanUseToolSlot(AttackToolBinding binding, bool reserveSilkForBind)
        {
            if (!PlayerData.HasInstance)
                return false;

            try
            {
                ToolItem tool = ToolItemManager.GetBoundAttackTool(binding, ToolEquippedReadSource.Active);
                if (!tool)
                    return false;

                int silk = PlayerData.instance.silk;
                if (tool.Type == ToolItemType.Skill)
                {
                    if (reserveSilkForBind)
                        return false;

                    int skillCost = Mathf.Max(1, PlayerData.instance.SilkSkillCost);
                    return silk >= skillCost;
                }

                if (tool.IsEmpty)
                    return false;

                return tool.Usage.SilkRequired <= 0 || silk >= tool.Usage.SilkRequired;
            }
            catch (Exception e)
            {
                RLManager.StaticLogger?.LogWarning($"[BossEncounterBase] Failed to build tool mask for {binding}: {e.Message}");
                return false;
            }
        }

        private bool IsSilkSpendingToolSlot(AttackToolBinding binding)
        {
            if (!PlayerData.HasInstance)
                return false;

            try
            {
                ToolItem tool = ToolItemManager.GetBoundAttackTool(binding, ToolEquippedReadSource.Active);
                if (!tool)
                    return false;

                return tool.Type == ToolItemType.Skill || tool.Usage.SilkRequired > 0;
            }
            catch (Exception e)
            {
                RLManager.StaticLogger?.LogWarning($"[BossEncounterBase] Failed to inspect tool slot {binding}: {e.Message}");
                return false;
            }
        }

        private void WriteRaycastObservations(float[] observations, HeroController hero, HealthManager boss)
        {
            int offset = CoreObservationSize + ResourceObservationSize;
            if (hero == null)
                return;

            Vector2 origin = hero.transform.position;
            RaycastHit2D[] hits = new RaycastHit2D[16];
            Collider2D[] virtualProjectileColliders = UnityEngine.Object.FindObjectsOfType<Collider2D>();
            for (int i = 0; i < RaycastCount; i++)
            {
                float angle = (360f / RaycastCount) * i * Mathf.Deg2Rad;
                Vector2 direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
                int hitCount = Physics2D.RaycastNonAlloc(origin, direction, hits, MaxRaycastDistance);
                float closestDistance = MaxRaycastDistance;
                int closestType = 0;

                for (int j = 0; j < hitCount; j++)
                {
                    RaycastHit2D hit = hits[j];
                    if (hit.collider == null || hit.distance <= 0.001f || hit.distance >= closestDistance)
                        continue;

                    if (hit.collider.GetComponentInParent<HeroController>() == hero)
                        continue;

                    int hitType = GetRaycastHitType(hit.collider, hero, boss);
                    if (hitType <= 0)
                        continue;

                    closestDistance = hit.distance;
                    closestType = hitType;
                }

                if (TryGetVirtualBossProjectileHit(
                        virtualProjectileColliders,
                        origin,
                        direction,
                        closestDistance,
                        out float virtualProjectileDistance))
                {
                    closestDistance = virtualProjectileDistance;
                    closestType = 5;
                }

                observations[offset + i] = Mathf.Clamp01(closestDistance / MaxRaycastDistance);
                int typeOffset = offset + RaycastCount + i * RaycastHitTypeCount;
                observations[typeOffset + Mathf.Clamp(closestType, 0, RaycastHitTypeCount - 1)] = 1f;
            }
        }

        private bool TryGetVirtualBossProjectileHit(
            Collider2D[] colliders,
            Vector2 origin,
            Vector2 direction,
            float currentClosestDistance,
            out float closestDistance)
        {
            closestDistance = currentClosestDistance;
            bool hit = false;
            if (colliders == null)
                return false;

            for (int i = 0; i < colliders.Length; i++)
            {
                Collider2D collider = colliders[i];
                if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy)
                    continue;

                string path = GetColliderPath(collider.transform).ToLowerInvariant();
                float radius = GetVirtualBossProjectileRadius(path);
                if (radius <= 0f)
                    continue;

                if (TryRayCircleDistance(origin, direction, collider.bounds.center, radius, closestDistance, out float distance))
                {
                    closestDistance = distance;
                    hit = true;
                }
            }

            return hit;
        }

        private float GetVirtualBossProjectileRadius(string path)
        {
            if (string.IsNullOrEmpty(path))
                return 0f;

            if (path.Contains("cross slash"))
                return LaceCrossSlashVirtualRadius;

            if (path.Contains("lace_circle_slash") ||
                path.Contains("circle slash") ||
                path.Contains("lace circle") ||
                path.Contains("multicircle"))
            {
                return LaceCircleSlashVirtualRadius;
            }

            return 0f;
        }

        private bool TryRayCircleDistance(
            Vector2 origin,
            Vector2 direction,
            Vector2 center,
            float radius,
            float maxDistance,
            out float distance)
        {
            Vector2 toCenter = center - origin;
            float along = Vector2.Dot(toCenter, direction);
            distance = 0f;
            if (along <= 0f || along > maxDistance + radius)
                return false;

            float perpendicularSqr = toCenter.sqrMagnitude - along * along;
            float radiusSqr = radius * radius;
            if (perpendicularSqr > radiusSqr)
                return false;

            float offset = Mathf.Sqrt(Mathf.Max(0f, radiusSqr - perpendicularSqr));
            distance = Mathf.Clamp(along - offset, 0.001f, MaxRaycastDistance);
            return distance < maxDistance;
        }

        private enum ObservedHitboxKind
        {
            None = 0,
            HeroHurtbox = 1,
            BossHurtbox = 2,
            BossAttack = 3,
            HeroAttack = 4,
            Hazard = 5,
            Other = 6
        }

        private void WriteHitboxObservations(float[] observations, HeroController hero, HealthManager boss)
        {
            int offset = CoreObservationSize + ResourceObservationSize + RaycastObservationSize;
            if (hero == null)
                return;

            Vector2 heroPosition = hero.transform.position;
            Collider2D[] colliders = UnityEngine.Object.FindObjectsOfType<Collider2D>();
            Collider2D[] selectedColliders = new Collider2D[HitboxObservationCount];
            ObservedHitboxKind[] selectedKinds = new ObservedHitboxKind[HitboxObservationCount];
            float[] selectedScores = new float[HitboxObservationCount];
            for (int i = 0; i < selectedScores.Length; i++)
            {
                selectedScores[i] = float.NegativeInfinity;
            }

            for (int i = 0; i < colliders.Length; i++)
            {
                Collider2D collider = colliders[i];
                if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy)
                    continue;

                ObservedHitboxKind kind = GetObservedHitboxKind(collider, hero, boss);
                if (kind == ObservedHitboxKind.None)
                    continue;

                Bounds bounds = collider.bounds;
                Vector2 center = bounds.center;
                float distance = Vector2.Distance(heroPosition, center);
                if (distance > ArenaDiagonal && kind != ObservedHitboxKind.BossAttack)
                    continue;

                float score = GetHitboxPriority(kind) - distance - bounds.size.x * 0.01f - bounds.size.y * 0.01f;
                for (int slot = 0; slot < HitboxObservationCount; slot++)
                {
                    if (score <= selectedScores[slot])
                        continue;

                    for (int shift = HitboxObservationCount - 1; shift > slot; shift--)
                    {
                        selectedScores[shift] = selectedScores[shift - 1];
                        selectedColliders[shift] = selectedColliders[shift - 1];
                        selectedKinds[shift] = selectedKinds[shift - 1];
                    }

                    selectedScores[slot] = score;
                    selectedColliders[slot] = collider;
                    selectedKinds[slot] = kind;
                    break;
                }
            }

            for (int slot = 0; slot < HitboxObservationCount; slot++)
            {
                Collider2D collider = selectedColliders[slot];
                if (collider == null)
                    continue;

                Bounds bounds = collider.bounds;
                int featureOffset = offset + slot * HitboxFeatureSize;
                observations[featureOffset] = 1f;
                observations[featureOffset + 1] = NormalizeSigned(bounds.center.x - heroPosition.x, ArenaWidth);
                observations[featureOffset + 2] = NormalizeSigned(bounds.center.y - heroPosition.y, ArenaHeight);
                observations[featureOffset + 3] = Mathf.Clamp01(bounds.size.x / MaxHitboxObservedWidth);
                observations[featureOffset + 4] = Mathf.Clamp01(bounds.size.y / MaxHitboxObservedHeight);

                int typeIndex = Mathf.Clamp((int)selectedKinds[slot] - 1, 0, HitboxTypeCount - 1);
                observations[featureOffset + 5 + typeIndex] = 1f;
            }
        }

        private ObservedHitboxKind GetObservedHitboxKind(Collider2D collider, HeroController hero, HealthManager boss)
        {
            if (collider == null)
                return ObservedHitboxKind.None;

            string path = GetColliderPath(collider.transform).ToLowerInvariant();
            if (IsIgnoredRangeCollider(path) || IsStaticStageHazardCollider(path))
                return ObservedHitboxKind.None;

            if (hero != null && collider.GetComponentInParent<HeroController>() == hero)
            {
                return IsHeroAttackCollider(path)
                    ? ObservedHitboxKind.HeroAttack
                    : ObservedHitboxKind.HeroHurtbox;
            }

            if (IsStandaloneBossAttackCollider(collider, path))
                return ObservedHitboxKind.BossAttack;

            if (boss != null && collider.GetComponentInParent<HealthManager>() == boss)
            {
                if (IsBossAttackCollider(collider, path))
                    return ObservedHitboxKind.BossAttack;

                if (collider.GetComponent<HealthManager>() == boss ||
                    collider.transform == boss.transform ||
                    collider.gameObject.layer == 11 ||
                    collider.gameObject.layer == 2)
                {
                    return ObservedHitboxKind.BossHurtbox;
                }

                return ObservedHitboxKind.None;
            }

            if (collider.GetComponent<DamageHero>() != null || collider.GetComponentInParent<DamageHero>() != null)
                return ObservedHitboxKind.Hazard;

            return ObservedHitboxKind.None;
        }

        private bool IsBossAttackCollider(Collider2D collider, string path)
        {
            if (collider == null)
                return false;

            if (collider.gameObject.layer == 22)
                return true;

            if (collider.GetComponent<DamageHero>() != null || collider.GetComponentInParent<DamageHero>() != null)
                return true;

            return path.Contains(" hit") ||
                path.Contains("/hit") ||
                path.Contains("slash") ||
                path.Contains("multihit") ||
                path.Contains("bodycatcher") ||
                path.Contains("damager") ||
                path.Contains("projectile");
        }

        private bool IsStandaloneBossAttackCollider(Collider2D collider, string path)
        {
            if (collider == null)
                return false;

            if (collider.gameObject.layer == 22)
                return true;

            return path.Contains("cross slash") ||
                path.Contains("lace_circle_slash") ||
                path.Contains("circle slash") ||
                path.Contains("lace circle") ||
                path.Contains("multicircle");
        }

        private bool IsHeroAttackCollider(string path)
        {
            return path.Contains("/attacks/") ||
                path.Contains("/slash") ||
                path.Contains("/upslash") ||
                path.Contains("/downslash");
        }

        private bool IsIgnoredRangeCollider(string path)
        {
            return path.Contains("nothwip range") ||
                path.Contains("evade range") ||
                path.Contains("above range") ||
                path.Contains("wall range") ||
                path.Contains("camera") ||
                path.Contains("enviro region") ||
                path.Contains("hazard respawn trigger");
        }

        private bool IsStaticStageHazardCollider(string path)
        {
            return path.Contains("steam hazard") ||
                path.Contains("steam damage collider");
        }

        private float GetHitboxPriority(ObservedHitboxKind kind)
        {
            switch (kind)
            {
                case ObservedHitboxKind.BossAttack:
                    return 1000f;
                case ObservedHitboxKind.BossHurtbox:
                    return 800f;
                case ObservedHitboxKind.HeroAttack:
                    return 700f;
                case ObservedHitboxKind.Hazard:
                    return 600f;
                case ObservedHitboxKind.HeroHurtbox:
                    return 500f;
                default:
                    return 100f;
            }
        }

        private string GetColliderPath(Transform transform)
        {
            if (transform == null)
                return "";

            string path = transform.name;
            Transform parent = transform.parent;
            while (parent != null)
            {
                path = parent.name + "/" + path;
                parent = parent.parent;
            }

            return path;
        }

        private int GetRaycastHitType(Collider2D collider, HeroController hero, HealthManager boss)
        {
            if (collider == null)
                return 0;

            if (collider.GetComponent<DamageHero>() != null ||
                collider.GetComponentInParent<DamageHero>() != null)
            {
                return 4;
            }

            HealthManager healthManager = collider.GetComponent<HealthManager>() ?? collider.GetComponentInParent<HealthManager>();
            if (healthManager != null)
            {
                return healthManager == boss ? 2 : 2;
            }

            string objectName = collider.gameObject.name.ToLowerInvariant();
            string parentName = collider.transform.parent != null
                ? collider.transform.parent.name.ToLowerInvariant()
                : "";
            if (objectName.Contains("damager") ||
                objectName.Contains("projectile") ||
                objectName.Contains("slash") ||
                parentName.Contains("slash") ||
                parentName.Contains("projectile") ||
                parentName.Contains("lace_circle"))
            {
                return 5;
            }

            int layer = collider.gameObject.layer;
            if (layer == 12)
                return 3;

            if (IsEnvironmentCollider(collider, hero, boss))
                return 1;

            return 0;
        }

        private bool IsEnvironmentCollider(Collider2D collider, HeroController hero, HealthManager boss)
        {
            if (collider == null || collider.isTrigger)
                return false;

            if (hero != null && collider.GetComponentInParent<HeroController>() == hero)
                return false;

            if (boss != null && collider.GetComponentInParent<HealthManager>() == boss)
                return false;

            Rigidbody2D body = collider.attachedRigidbody;
            if (body != null && body.bodyType == RigidbodyType2D.Dynamic)
                return false;

            if (collider.GetComponentInParent<HealthManager>() != null)
                return false;

            return true;
        }

        private float GetToolAmountLoss(float[] previousObs, float[] currentObs)
        {
            float loss = 0f;
            for (int i = 0; i < 3; i++)
            {
                int amountIndex = ToolSlotStartIndex + i * ToolSlotSize + 2;
                loss += Mathf.Max(0f, previousObs[amountIndex] - currentObs[amountIndex]);
            }
            return loss;
        }

        private float NormalizePositionX(float value)
        {
            return Mathf.Clamp01((value - MinPosX) / ArenaWidth);
        }

        private float DenormalizePositionX(float value)
        {
            return value * ArenaWidth + MinPosX;
        }

        private float NormalizePositionY(float value)
        {
            return Mathf.Clamp01((value - MinPosY) / ArenaHeight);
        }

        private float DenormalizePositionY(float value)
        {
            return value * ArenaHeight + MinPosY;
        }

        private float NormalizeSigned(float value, float maxAbs)
        {
            if (maxAbs <= 0f)
                return 0.5f;

            return Mathf.Clamp01((value + maxAbs) / (2f * maxAbs));
        }

        private float DenormalizeSigned(float value, float maxAbs)
        {
            if (maxAbs <= 0f)
                return 0f;

            return (Mathf.Clamp01(value) * 2f - 1f) * maxAbs;
        }

        private float ArenaWidth => Mathf.Max(0.001f, MaxPosX - MinPosX);
        private float ArenaHeight => Mathf.Max(0.001f, MaxPosY - MinPosY);
        private float ArenaDiagonal => Mathf.Sqrt(ArenaWidth * ArenaWidth + ArenaHeight * ArenaHeight);
    }
}
