using System;
using HarmonyLib;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;

namespace SilksongRL
{
    public enum MoveDirection { None = 0, Left = 1, Right = 2 }
    public enum LookDirection { None = 0, Up = 1, Down = 2 }
    public enum ToolAction { None = 0, Neutral = 1, Up = 2, Down = 3 }
    public enum OffenseAction { None = 0, HorizontalSlash = 1, AntiAirSlash = 2, RangedTool = 3 }

    public enum LowLevelAction
    {
        Idle = 0,
        MoveLeft = 1,
        MoveRight = 2,
        LookUp = 3,
        LookDown = 4,
        Jump = 5,
        JumpLeft = 6,
        JumpRight = 7,
        Dash = 8,
        DashLeft = 9,
        DashRight = 10,
        Attack = 11,
        MoveLeftAttack = 12,
        MoveRightAttack = 13,
        UpAttack = 14,
        DownAttack = 15,
        JumpAttack = 16,
        JumpLeftAttack = 17,
        JumpRightAttack = 18,
        DashLeftAttack = 19,
        DashRightAttack = 20,
        Needle = 21,
        BindHeal = 22,
        ToolNeutral = 23,
        ToolUp = 24,
        ToolDown = 25,
        MoveLeftTool = 26,
        MoveRightTool = 27,
        JumpLeftTool = 28,
        JumpRightTool = 29
    }

    public enum MacroAction
    {
        KeepSweetSpot = 0,
        RetreatFromBossDodge = 1,
        DodgeOverBossHighJump = 2,
        CrossUnderAirBoss = 3,
        MoveToCenterSafely = 4,
        HoldSafeSide = 5,
        KeepSweetSpotSlash = 6,
        RetreatSlash = 7,
        DodgeOverBossSlash = 8,
        CrossUnderAirBossSlash = 9,
        MoveToCenterSlash = 10,
        PunishWithSlash = 11,
        KeepSweetSpotAntiAir = 12,
        RetreatAntiAir = 13,
        DodgeOverBossAntiAir = 14,
        CrossUnderAirBossAntiAir = 15,
        MoveToCenterAntiAir = 16,
        AntiAirSlash = 17,
        KeepSweetSpotRangedTool = 18,
        RetreatRangedTool = 19,
        DodgeOverBossRangedTool = 20,
        CrossUnderAirBossRangedTool = 21,
        MoveToCenterRangedTool = 22,
        UseRangedTool = 23,
        UseCloseSkill = 24,
        BindHeal = 25
    }

    public class Action
    {
        public MacroAction macro;
        public int actionIndex;
        public bool optionStarted;
        public MoveDirection move;
        public LookDirection look;
        public bool jump;
        public bool attack;
        public bool dash;
        public bool needle;
        public bool bind;
        public ToolAction tool;

        public Action()
        {
            macro = MacroAction.HoldSafeSide;
            actionIndex = (int)LowLevelAction.Idle;
            optionStarted = false;
            move = MoveDirection.None;
            look = LookDirection.None;
            jump = false;
            attack = false;
            dash = false;
            needle = false;
            bind = false;
            tool = ToolAction.None;
        }
    }

    /// <summary>
    /// Manages the public macro action space and the internal key-level action representation.
    /// </summary>
    public static class ActionManager
    {
        public const int MovementMacroCount = 6;
        public const int OffenseActionCount = 4;
        public const int CompositeActionCount = MovementMacroCount * OffenseActionCount;
        public const int MacroActionCount = CompositeActionCount + 2;
        public const int LowLevelActionCount = 30;
        public const int MacroOffset = 0;
        private static readonly bool UseRawActionBranches = false;

        public const int MoveOffset = 0;
        public const int LookOffset = 3;
        public const int JumpOffset = 6;
        public const int AttackOffset = 8;
        public const int DashOffset = 10;
        public const int NeedleOffset = 12;
        public const int BindOffset = 14;
        public const int ToolOffset = 16;

        private const float KeepRangeNear = 3.2f;//本来1.55
        private const float KeepRangeFar = 4.7f;//本来3.5
        private const float CollisionBufferRange = 0.95f;
        private const float CenterDeadzone = 0.45f;
        private const float SlashRetreatRange = 0.8f;
        private const float SlashStepTowardRange = 1.7f;
        private const float JumpDangerRange = 4.0f;
        private const float DashThroughDangerRange = 4.0f;
        private const float VerticalAttackHorizontalToleranceFallback = 1.5f;
        private const float RetreatDashDangerRange = 4.5f;
        private const float RetreatDashPanicRange = 2.2f;
        private const float RetreatDashVerticalTolerance = 2.2f;
        private const float DodgeIncomingHoldHorizontalRange = 4.2f;
        private const float DodgeIncomingHoldVerticalTolerance = 2.2f;
        private const float DodgeIncomingMinBossSpeed = 4.0f;
        private const float DodgeOverBossMinHeroAboveBoss = 2.5f;
        private const float DodgeOverBossMinJumpHoldSeconds = 0.70f;
        private const float DodgeOverBossJumpReleaseGapSeconds = 0.05f;
        private const float DodgeOverBossAirJumpHoldSeconds = 0.18f;
        private const float DodgeOverBossPassClearance = 1.35f;
        private const float DodgeAirDashHorizontalRange = 3.4f;
        private const float DodgeAirDashVerticalTolerance = 2.3f;
        private const float OptionDurationJitterSeconds = 0.05f;
        private const float BindHealOptionSeconds = 0.85f;
        private const float DodgeOverBossHighJumpOptionSeconds = 1.05f;
        private const float CrossUnderAirBossOptionSeconds = 0.40f;
        private const float RetreatFromBossDodgeOptionSeconds = 0.36f;
        private const float MoveToCenterSafelyOptionSeconds = 0.20f;
        private const float BasicAttackTapSeconds = 0.055f;
        private const float BasicAttackTapIntervalSeconds = 0.16f;

        private static float bindHealOptionUntil = -1f;
        private static float defensiveOptionUntil = -1f;
        private static float defensiveOptionStartedAt = -1f;
        private static MacroAction defensiveSustainedMacro = MacroAction.HoldSafeSide;
        private static float dodgeOverBossJumpCycleStartedAt = -1f;
        private static float dodgeOverBossAirJumpHoldUntil = -1f;
        private static MoveDirection dodgeOverBossPassDirection = MoveDirection.None;
        private static bool dodgeOverBossWasAirborne = false;
        private static bool dodgeOverBossAirJumpUsed = false;
        private static float attackTapPressUntil = -1f;
        private static float nextAttackTapStartAt = -1f;

        public static int[] GetActionSpaceShape()
        {
            if (UseRawActionBranches)
                return GetRawActionSpaceShape();

            return new int[] { MacroActionCount };
        }

        public static int[] GetRawActionSpaceShape()
        {
            return new int[]
            {
                3, // move: none, left, right
                3, // look: none, up, down
                2, // jump
                2, // attack
                2, // dash
                2, // needle tether (S)
                2, // bind/heal (A)
                4  // tool/skill: none, F, F+Up, F+Down
            };
        }

        public static int GetActionMaskSize()
        {
            if (UsesRawActionSpace())
                return GetRawActionMaskSize();

            return GetJointActionCount(GetActionSpaceShape());
        }

        public static int GetRawActionMaskSize()
        {
            return GetMaskSize(GetRawActionSpaceShape());
        }

        public static int[] CreateAllValidActionMask()
        {
            return CreateAllValidActionMask(GetActionMaskSize());
        }

        public static int[] CreateRawAllValidActionMask()
        {
            return CreateAllValidActionMask(GetRawActionMaskSize());
        }

        private static int[] CreateAllValidActionMask(int size)
        {
            int[] mask = new int[size];
            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = 1;
            }

            return mask;
        }

        public static void DisableActionValue(int[] mask, int offset, int value)
        {
            if (mask == null)
                return;

            int index = offset + value;
            if (index >= 0 && index < mask.Length)
            {
                mask[index] = 0;
            }
        }

        public static void EnableActionValue(int[] mask, int offset, int value)
        {
            if (mask == null)
                return;

            int index = offset + value;
            if (index >= 0 && index < mask.Length)
            {
                mask[index] = 1;
            }
        }

        public static bool IsActionValueEnabled(int[] mask, int offset, int value)
        {
            if (mask == null)
                return true;

            int index = offset + value;
            return index >= 0 && index < mask.Length && mask[index] != 0;
        }

        public static void EnsureValidChoices(int[] mask)
        {
            if (mask == null)
                return;

            if (mask.Length == GetJointActionCount(GetActionSpaceShape()))
            {
                bool hasValidChoice = false;
                for (int i = 0; i < mask.Length; i++)
                {
                    if (mask[i] != 0)
                    {
                        hasValidChoice = true;
                        break;
                    }
                }

                if (!hasValidChoice && mask.Length > 0)
                {
                    mask[0] = 1;
                }

                return;
            }

            EnsureValidChoices(mask, GetActionSpaceShape());
        }

        public static void EnsureRawValidChoices(int[] mask)
        {
            EnsureValidChoices(mask, GetRawActionSpaceShape());
        }

        private static void EnsureValidChoices(int[] mask, int[] shape)
        {
            if (mask == null)
                return;

            int offset = 0;
            for (int i = 0; i < shape.Length; i++)
            {
                bool hasValidChoice = false;
                for (int j = 0; j < shape[i]; j++)
                {
                    if (mask[offset + j] != 0)
                    {
                        hasValidChoice = true;
                        break;
                    }
                }

                if (!hasValidChoice)
                {
                    mask[offset] = 1;
                }

                offset += shape[i];
            }
        }

        private static int GetMaskSize(int[] shape)
        {
            int size = 0;
            for (int i = 0; i < shape.Length; i++)
            {
                size += shape[i];
            }

            return size;
        }

        private static int GetJointActionCount(int[] shape)
        {
            int count = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                count *= Mathf.Max(1, shape[i]);
            }

            return count;
        }

        public static Action ArrayToAction(ActionResponse response)
        {
            if (response?.action == null)
            {
                RLManager.StaticLogger?.LogError("[ActionManager] Invalid action response received");
                return new Action();
            }

            if (response.action.Length == 1)
            {
                return ActionIndexToAction(response.action[0], RLManager.Hero, RLManager.Boss, RLManager.CurrentEncounter);
            }

            int rawSize = GetRawActionSpaceShape().Length;
            if (response.action.Length == rawSize)
            {
                return RawArrayToAction(response.action, RLManager.Hero, RLManager.Boss, RLManager.CurrentEncounter);
            }

            RLManager.StaticLogger?.LogError($"[ActionManager] Action array size mismatch. Expected 1 or {rawSize}, got {response.action.Length}");
            return new Action();
        }

        public static int GetActionCount()
        {
            return GetJointActionCount(GetActionSpaceShape());
        }

        public static Action ActionIndexToAction(int actionIndex, HeroController hero, HealthManager boss, IBossEncounter encounter)
        {
            if (UsesMacroActionSpace())
            {
                return MacroToRawAction(ClampMacroAction(actionIndex), hero, boss, encounter);
            }

            return RawArrayToAction(ActionIndexToRawArray(actionIndex), hero, boss, encounter);
        }

        public static int[] TeacherMacroToActionArray(
            MacroAction teacherMacro,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            return ActionToArray(MacroToRawAction(teacherMacro, hero, boss, encounter));
        }

        public static string GetActionName(Action action)
        {
            if (action == null)
                return "Idle";

            return $"{action.move}/{action.look}/J{(action.jump ? 1 : 0)}/A{(action.attack ? 1 : 0)}/D{(action.dash ? 1 : 0)}/N{(action.needle ? 1 : 0)}/B{(action.bind ? 1 : 0)}/T{action.tool}";
        }

        public static void PrepareActionForExecution(Action action)
        {
            if (action == null)
                return;

            bool forceBasicAttack =
                RLManager.CurrentEncounter != null &&
                RLManager.CurrentEncounter.ShouldForceBasicAttack() &&
                IsBasicAttackTapActiveOrReady() &&
                !action.bind &&
                !action.needle &&
                action.tool == ToolAction.None &&
                action.macro != MacroAction.UseCloseSkill &&
                action.macro != MacroAction.BindHeal &&
                (RLManager.Hero == null || (!RLManager.Hero.cState.dead && !RLManager.Hero.cState.recoiling));

            if (forceBasicAttack)
            {
                action.attack = true;
            }

            bool wantsBasicAttack =
                action.attack ||
                IsHorizontalSlashAction(action.macro) ||
                IsAntiAirSlashAction(action.macro);

            if (!wantsBasicAttack ||
                action.bind ||
                action.needle ||
                action.tool != ToolAction.None ||
                action.macro == MacroAction.UseCloseSkill ||
                action.macro == MacroAction.BindHeal)
            {
                action.attack = false;
                return;
            }

            action.attack = ShouldEmitBasicAttackTap();
            if (action.attack && action.look == LookDirection.None)
            {
                FaceBossForHorizontalSlash(action, RLManager.Hero, RLManager.Boss);
            }
        }

        public static bool IsBasicAttackTapActiveOrReady()
        {
            float now = Time.time;
            return now <= attackTapPressUntil || now >= nextAttackTapStartAt;
        }

        public static Action MacroToRawAction(
            MacroAction macro,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            Action action = new Action
            {
                macro = macro,
                actionIndex = (int)macro
            };

            if (macro == MacroAction.UseCloseSkill)
            {
                action.tool = ToolAction.Neutral;
                FaceBossIfNeeded(action, hero, boss);
                ClampMovementToSafeArea(action, hero, encounter);
                action.actionIndex = ActionToIndex(action);
                return action;
            }

            if (macro == MacroAction.BindHeal)
            {
                action.bind = true;
                ClampMovementToSafeArea(action, hero, encounter);
                action.actionIndex = ActionToIndex(action);
                return action;
            }

            MacroAction movementMacro = GetMovementMacro(macro);

            switch (movementMacro)
            {
                case MacroAction.KeepSweetSpot:
                    action.move = GetKeepRangeDirection(hero, boss);
                    break;
                case MacroAction.RetreatFromBossDodge:
                    action.move = GetRetreatFromBossDodgeDirection(hero, boss, encounter);
                    action.dash = ShouldDashRetreat(hero, boss, encounter);
                    action.jump = ShouldJumpRetreat(hero, boss);
                    break;
                case MacroAction.DodgeOverBossHighJump:
                    action.move = GetDodgeOverBossMoveDirection(hero, boss, encounter);
                    action.jump = true;
                    if (ShouldAirDashThrough(hero, boss, encounter, requireCompletedHighJump: true))
                    {
                        action.move = GetDodgeThroughDirection(hero, boss, encounter);
                        action.jump = false;
                        action.dash = true;
                    }
                    ApplyDodgeOverBossJumpBudget(action, hero, boss, encounter);
                    break;
                case MacroAction.CrossUnderAirBoss:
                    action.move = GetCrossUnderAirBossDirection(hero, boss, encounter);
                    action.dash = ShouldDashCrossUnder(hero, boss);
                    break;
                case MacroAction.MoveToCenterSafely:
                    if (ShouldJumpOverBossToReachCenter(hero, boss, encounter))
                    {
                        action.move = GetDodgeOverBossMoveDirection(hero, boss, encounter);
                        action.jump = true;
                        if (ShouldAirDashThrough(hero, boss, encounter, requireCompletedHighJump: true))
                        {
                            action.move = GetDodgeThroughDirection(hero, boss, encounter);
                            action.jump = false;
                            action.dash = true;
                        }
                        ApplyDodgeOverBossJumpBudget(action, hero, boss, encounter);
                    }
                    else
                    {
                        action.move = GetSafeCenterDirection(hero, boss, encounter);
                    }
                    break;
                case MacroAction.HoldSafeSide:
                    action.move = GetHoldSafeSideDirection(hero, boss, encounter);
                    break;
            }

            ApplyOffenseOverlay(action, GetOffenseAction(macro), hero, boss, encounter);
            ClampMovementToSafeArea(action, hero, encounter);
            NormalizeRawActionConflicts(action);
            MacroAction effectiveMacro = encounter != null && encounter.ShouldForceBasicAttack()
                ? ForceBasicAttackUnlessSpecial(action, macro, hero, boss)
                : macro;
            NormalizeRawActionConflicts(action);
            action.macro = effectiveMacro;
            action.actionIndex = ActionToIndex(action);
            return action;
        }

        private static Action LowLevelToRawAction(
            LowLevelAction lowLevelAction,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            Action action = new Action
            {
                actionIndex = (int)lowLevelAction
            };

            switch (lowLevelAction)
            {
                case LowLevelAction.MoveLeft:
                    action.move = MoveDirection.Left;
                    break;
                case LowLevelAction.MoveRight:
                    action.move = MoveDirection.Right;
                    break;
                case LowLevelAction.LookUp:
                    action.look = LookDirection.Up;
                    break;
                case LowLevelAction.LookDown:
                    action.look = LookDirection.Down;
                    break;
                case LowLevelAction.Jump:
                    action.jump = true;
                    break;
                case LowLevelAction.JumpLeft:
                    action.move = MoveDirection.Left;
                    action.jump = true;
                    break;
                case LowLevelAction.JumpRight:
                    action.move = MoveDirection.Right;
                    action.jump = true;
                    break;
                case LowLevelAction.Dash:
                    action.dash = true;
                    break;
                case LowLevelAction.DashLeft:
                    action.move = MoveDirection.Left;
                    action.dash = true;
                    break;
                case LowLevelAction.DashRight:
                    action.move = MoveDirection.Right;
                    action.dash = true;
                    break;
                case LowLevelAction.Attack:
                    action.attack = true;
                    FaceBossIfNeeded(action, hero, boss);
                    break;
                case LowLevelAction.MoveLeftAttack:
                    action.move = MoveDirection.Left;
                    action.attack = true;
                    break;
                case LowLevelAction.MoveRightAttack:
                    action.move = MoveDirection.Right;
                    action.attack = true;
                    break;
                case LowLevelAction.UpAttack:
                    action.look = LookDirection.Up;
                    action.attack = true;
                    break;
                case LowLevelAction.DownAttack:
                    action.look = LookDirection.Down;
                    action.attack = true;
                    break;
                case LowLevelAction.JumpAttack:
                    action.jump = true;
                    action.attack = true;
                    FaceBossIfNeeded(action, hero, boss);
                    break;
                case LowLevelAction.JumpLeftAttack:
                    action.move = MoveDirection.Left;
                    action.jump = true;
                    action.attack = true;
                    break;
                case LowLevelAction.JumpRightAttack:
                    action.move = MoveDirection.Right;
                    action.jump = true;
                    action.attack = true;
                    break;
                case LowLevelAction.DashLeftAttack:
                    action.move = MoveDirection.Left;
                    action.dash = true;
                    action.attack = true;
                    break;
                case LowLevelAction.DashRightAttack:
                    action.move = MoveDirection.Right;
                    action.dash = true;
                    action.attack = true;
                    break;
                case LowLevelAction.Needle:
                    action.needle = true;
                    FaceBossIfNeeded(action, hero, boss);
                    break;
                case LowLevelAction.BindHeal:
                    action.bind = true;
                    break;
                case LowLevelAction.ToolNeutral:
                    action.tool = ToolAction.Neutral;
                    FaceBossIfNeeded(action, hero, boss);
                    break;
                case LowLevelAction.ToolUp:
                    action.tool = ToolAction.Up;
                    FaceBossIfNeeded(action, hero, boss);
                    break;
                case LowLevelAction.ToolDown:
                    action.tool = ToolAction.Down;
                    FaceBossIfNeeded(action, hero, boss);
                    break;
                case LowLevelAction.MoveLeftTool:
                    action.move = MoveDirection.Left;
                    action.tool = ToolAction.Neutral;
                    break;
                case LowLevelAction.MoveRightTool:
                    action.move = MoveDirection.Right;
                    action.tool = ToolAction.Neutral;
                    break;
                case LowLevelAction.JumpLeftTool:
                    action.move = MoveDirection.Left;
                    action.jump = true;
                    action.tool = ToolAction.Neutral;
                    break;
                case LowLevelAction.JumpRightTool:
                    action.move = MoveDirection.Right;
                    action.jump = true;
                    action.tool = ToolAction.Neutral;
                    break;
            }

            action.macro = InferMacroAction(action, hero, boss, encounter);
            ClampMovementToSafeArea(action, hero, encounter);
            action.actionIndex = ActionToIndex(action);
            return action;
        }

        private static void ApplyOffenseOverlay(
            Action action,
            OffenseAction offense,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (action == null)
                return;

            switch (offense)
            {
                case OffenseAction.HorizontalSlash:
                    action.attack = true;
                    FaceBossForHorizontalSlash(action, hero, boss);
                    break;
                case OffenseAction.AntiAirSlash:
                    if (action.move == MoveDirection.None)
                    {
                        action.move = GetAntiAirPositionDirection(hero, boss, encounter);
                    }
                    action.look = LookDirection.Up;
                    action.attack = true;
                    break;
                case OffenseAction.RangedTool:
                    action.tool = GetRangedToolAction(hero, boss);
                    FaceBossIfStationary(action, hero, boss);
                    break;
            }
        }

        public static void BeginSustainedOption(Action action)
        {
            if (action == null)
                return;

            if (action.macro == MacroAction.BindHeal)
            {
                action.optionStarted = true;
                bindHealOptionUntil = Mathf.Max(bindHealOptionUntil, Time.fixedTime + BindHealOptionSeconds);
                return;
            }

            if (UsesRawActionSpace())
                return;

            if (IsDefensiveCommitMacro(action.macro))
            {
                action.optionStarted = true;
                defensiveSustainedMacro = action.macro;
                defensiveOptionStartedAt = Time.fixedTime;
                if (GetMovementMacro(action.macro) == MacroAction.DodgeOverBossHighJump)
                {
                    StartDodgeOverBossJumpCycle(defensiveOptionStartedAt);
                }
                else
                {
                    ResetDodgeOverBossJumpCycle();
                }
                defensiveOptionUntil = Mathf.Max(
                    defensiveOptionUntil,
                    Time.fixedTime + GetSustainedOptionSeconds(action.macro));
            }
        }

        public static bool ShouldBeginSustainedOption(Action action)
        {
            if (action == null)
                return false;

            if (action.macro == MacroAction.BindHeal || action.bind)
                return true;

            return !UsesRawActionSpace() && IsDefensiveCommitMacro(action.macro);
        }

        public static bool TryGetSustainedAction(HeroController hero, HealthManager boss, IBossEncounter encounter, out Action action)
        {
            action = null;

            if (hero == null || hero.cState.dead || hero.cState.recoiling)
            {
                CancelSustainedOption();
                return false;
            }

            if (!UsesRawActionSpace() &&
                Time.fixedTime <= defensiveOptionUntil &&
                IsDefensiveCommitMacro(defensiveSustainedMacro))
            {
                if (TryGetDefensiveInterrupt(hero, boss, encounter, defensiveSustainedMacro, out MacroAction interruptMacro))
                {
                    defensiveSustainedMacro = interruptMacro;
                    defensiveOptionStartedAt = Time.fixedTime;
                    if (GetMovementMacro(interruptMacro) == MacroAction.DodgeOverBossHighJump)
                    {
                        StartDodgeOverBossJumpCycle(defensiveOptionStartedAt);
                    }
                    else
                    {
                        ResetDodgeOverBossJumpCycle();
                    }
                    defensiveOptionUntil = Time.fixedTime + GetSustainedOptionSeconds(interruptMacro);
                }

                if (ShouldCancelDefensiveOption(hero, boss, encounter, defensiveSustainedMacro))
                {
                    defensiveOptionUntil = -1f;
                    defensiveSustainedMacro = MacroAction.HoldSafeSide;
                    ResetDodgeOverBossJumpCycle();
                    return false;
                }

                action = MacroToRawAction(defensiveSustainedMacro, hero, boss, encounter);
                action.optionStarted = false;
                return true;
            }

            defensiveOptionUntil = -1f;
            defensiveSustainedMacro = MacroAction.HoldSafeSide;
            defensiveOptionStartedAt = -1f;

            if (Time.fixedTime > bindHealOptionUntil)
            {
                bindHealOptionUntil = -1f;
                return false;
            }

            if (hero.playerData != null &&
                hero.playerData.health >= Mathf.Max(1, hero.playerData.CurrentMaxHealth))
            {
                CancelSustainedOption();
                return false;
            }

            action = MacroToRawAction(MacroAction.BindHeal, hero, boss, encounter);
            action.optionStarted = false;
            return true;
        }

        public static void CancelSustainedOption()
        {
            bindHealOptionUntil = -1f;
            defensiveOptionUntil = -1f;
            defensiveSustainedMacro = MacroAction.HoldSafeSide;
            defensiveOptionStartedAt = -1f;
            ResetDodgeOverBossJumpCycle();
            attackTapPressUntil = -1f;
            nextAttackTapStartAt = -1f;
        }

        public static MacroAction GetMovementMacro(MacroAction macro)
        {
            int value = (int)macro;
            if (value >= 0 && value < CompositeActionCount)
            {
                return (MacroAction)(value % MovementMacroCount);
            }

            return macro;
        }

        public static OffenseAction GetOffenseAction(MacroAction macro)
        {
            int value = (int)macro;
            if (value >= 0 && value < CompositeActionCount)
            {
                return (OffenseAction)(value / MovementMacroCount);
            }

            return OffenseAction.None;
        }

        public static MacroAction ComposeMacroAction(MacroAction movementMacro, OffenseAction offense)
        {
            if (movementMacro == MacroAction.UseCloseSkill || movementMacro == MacroAction.BindHeal)
                return movementMacro;

            int movement = Mathf.Clamp((int)GetMovementMacro(movementMacro), 0, MovementMacroCount - 1);
            int offenseIndex = Mathf.Clamp((int)offense, 0, OffenseActionCount - 1);
            return (MacroAction)(offenseIndex * MovementMacroCount + movement);
        }

        public static bool IsHorizontalSlashAction(MacroAction macro)
        {
            return GetOffenseAction(macro) == OffenseAction.HorizontalSlash;
        }

        public static bool IsAntiAirSlashAction(MacroAction macro)
        {
            return GetOffenseAction(macro) == OffenseAction.AntiAirSlash;
        }

        public static bool IsRangedToolAction(MacroAction macro)
        {
            return GetOffenseAction(macro) == OffenseAction.RangedTool;
        }

        public static bool IsCloseSkillAction(MacroAction macro)
        {
            return macro == MacroAction.UseCloseSkill;
        }

        public static bool IsBindHealAction(MacroAction macro)
        {
            return macro == MacroAction.BindHeal;
        }

        private static OffenseAction InferOffenseAction(Action rawAction)
        {
            if (rawAction == null)
                return OffenseAction.None;

            if (rawAction.tool == ToolAction.Up || rawAction.tool == ToolAction.Down || rawAction.needle)
                return OffenseAction.RangedTool;

            if (rawAction.attack)
                return rawAction.look == LookDirection.Up ? OffenseAction.AntiAirSlash : OffenseAction.HorizontalSlash;

            return OffenseAction.None;
        }

        public static MacroAction InferMacroAction(
            Action rawAction,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (rawAction == null)
                return MacroAction.HoldSafeSide;

            if (rawAction.bind)
                return MacroAction.BindHeal;

            if (rawAction.tool == ToolAction.Neutral)
                return MacroAction.UseCloseSkill;

            MacroAction movementMacro;
            if (rawAction.dash)
            {
                MoveDirection centerDirection = GetDirectionToCenter(hero, encounter);
                if (rawAction.move != MoveDirection.None &&
                    rawAction.move == centerDirection &&
                    IsFarFromCenter(hero, encounter))
                {
                    movementMacro = MacroAction.MoveToCenterSafely;
                }
                else
                {
                    MoveDirection towardBoss = GetDirectionToBoss(hero, boss);
                    movementMacro = rawAction.move != MoveDirection.None && rawAction.move == towardBoss
                        ? MacroAction.DodgeOverBossHighJump
                        : MacroAction.RetreatFromBossDodge;
                }

                return ComposeMacroAction(movementMacro, InferOffenseAction(rawAction));
            }

            if (rawAction.jump)
            {
                MoveDirection centerDirection = GetDirectionToCenter(hero, encounter);
                if (rawAction.move != MoveDirection.None &&
                    rawAction.move == centerDirection &&
                    IsFarFromCenter(hero, encounter))
                {
                    movementMacro = MacroAction.MoveToCenterSafely;
                }
                else
                {
                    MoveDirection towardBoss = GetDirectionToBoss(hero, boss);
                    if (rawAction.move == towardBoss)
                        movementMacro = MacroAction.DodgeOverBossHighJump;
                    else if (rawAction.move == Opposite(towardBoss))
                        movementMacro = MacroAction.RetreatFromBossDodge;
                    else
                        movementMacro = MacroAction.HoldSafeSide;
                }

                return ComposeMacroAction(movementMacro, InferOffenseAction(rawAction));
            }

            if (rawAction.move != MoveDirection.None)
            {
                MoveDirection centerDirection = GetDirectionToCenter(hero, encounter);
                if (centerDirection != MoveDirection.None && rawAction.move == centerDirection && IsFarFromCenter(hero, encounter))
                {
                    movementMacro = MacroAction.MoveToCenterSafely;
                }
                else
                {
                    MoveDirection towardBoss = GetDirectionToBoss(hero, boss);
                    movementMacro = rawAction.move == towardBoss ? MacroAction.KeepSweetSpot : MacroAction.RetreatFromBossDodge;
                }

                return ComposeMacroAction(movementMacro, InferOffenseAction(rawAction));
            }

            return ComposeMacroAction(MacroAction.HoldSafeSide, InferOffenseAction(rawAction));
        }

        public static int[] ActionToArray(Action action)
        {
            if (UsesMacroActionSpace())
            {
                return new int[] { ActionToIndex(action) };
            }

            return ActionToRawArray(action);
        }

        public static int[] ActionToRawArray(Action action)
        {
            if (action == null)
            {
                action = new Action();
            }

            return new int[]
            {
                (int)action.move,
                (int)action.look,
                action.jump ? 1 : 0,
                action.attack ? 1 : 0,
                action.dash ? 1 : 0,
                action.needle ? 1 : 0,
                action.bind ? 1 : 0,
                (int)action.tool
            };
        }

        public static int ActionToIndex(Action action)
        {
            if (UsesMacroActionSpace())
            {
                if (action == null)
                    return (int)MacroAction.HoldSafeSide;

                return Mathf.Clamp((int)action.macro, 0, MacroActionCount - 1);
            }

            return RawArrayToActionIndex(ActionToRawArray(action));
        }

        public static bool UsesMacroActionSpace()
        {
            int[] shape = GetActionSpaceShape();
            return shape.Length == 1 && shape[0] == MacroActionCount;
        }

        public static bool UsesRawActionSpace()
        {
            int[] shape = GetActionSpaceShape();
            int[] rawShape = GetRawActionSpaceShape();
            if (shape.Length != rawShape.Length)
                return false;

            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] != rawShape[i])
                    return false;
            }

            return true;
        }

        public static int RawArrayToActionIndex(int[] raw)
        {
            int[] shape = GetRawActionSpaceShape();
            int index = 0;
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                int value = 0;
                if (raw != null && i < raw.Length)
                {
                    value = Mathf.Clamp(raw[i], 0, shape[i] - 1);
                }

                index += value * stride;
                stride *= shape[i];
            }

            return Mathf.Clamp(index, 0, GetJointActionCount(shape) - 1);
        }

        public static int[] ActionIndexToRawArray(int actionIndex)
        {
            int[] shape = GetRawActionSpaceShape();
            int maxIndex = GetJointActionCount(shape) - 1;
            int value = Mathf.Clamp(actionIndex, 0, maxIndex);
            int[] raw = new int[shape.Length];
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                raw[i] = value % shape[i];
                value /= shape[i];
            }

            return raw;
        }

        private static Action RawArrayToAction(
            int[] raw,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            int[] values = NormalizeRawActionArray(raw);
            Action action = new Action
            {
                move = (MoveDirection)values[0],
                look = (LookDirection)values[1],
                jump = values[2] != 0,
                attack = values[3] != 0,
                dash = values[4] != 0,
                needle = values[5] != 0,
                bind = values[6] != 0,
                tool = (ToolAction)values[7]
            };

            NormalizeRawActionConflicts(action);
            action.macro = InferMacroAction(action, hero, boss, encounter);
            ClampMovementToSafeArea(action, hero, encounter);
            action.actionIndex = ActionToIndex(action);
            return action;
        }

        private static int[] NormalizeRawActionArray(int[] raw)
        {
            int[] shape = GetRawActionSpaceShape();
            int[] values = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                int value = raw != null && i < raw.Length ? raw[i] : 0;
                values[i] = Mathf.Clamp(value, 0, shape[i] - 1);
            }

            return values;
        }

        private static void NormalizeRawActionConflicts(Action action)
        {
            if (action == null)
                return;

            if (action.bind)
            {
                action.move = MoveDirection.None;
                action.look = LookDirection.None;
                action.jump = false;
                action.attack = false;
                action.dash = false;
                action.needle = false;
                action.tool = ToolAction.None;
                return;
            }

            if (action.tool != ToolAction.None)
            {
                action.attack = false;
                action.needle = false;
                if (action.tool == ToolAction.Up)
                {
                    action.look = LookDirection.Up;
                }
                else if (action.tool == ToolAction.Down)
                {
                    action.look = LookDirection.Down;
                }

                return;
            }

            if (action.needle)
            {
                action.attack = false;
            }

            if (action.look != LookDirection.None && action.tool == ToolAction.None)
            {
                // Looking up/down with horizontal movement is valid, but up/down slashes
                // should not also request a dash in the same frame.
                if (action.attack)
                {
                    action.dash = false;
                }
            }
        }

        private static MacroAction ForceBasicAttackUnlessSpecial(
            Action action,
            MacroAction selectedMacro,
            HeroController hero,
            HealthManager boss)
        {
            if (action == null)
                return selectedMacro;

            if (selectedMacro == MacroAction.UseCloseSkill ||
                selectedMacro == MacroAction.BindHeal ||
                action.bind ||
                action.needle ||
                action.tool != ToolAction.None)
            {
                return selectedMacro;
            }

            if (hero != null && (hero.cState.dead || hero.cState.recoiling))
                return selectedMacro;

            if (GetOffenseAction(selectedMacro) != OffenseAction.AntiAirSlash &&
                action.look == LookDirection.Down)
            {
                action.look = LookDirection.None;
            }

            action.attack = true;
            FaceBossIfStationary(action, hero, boss);

            OffenseAction offense = GetOffenseAction(selectedMacro);
            if (offense == OffenseAction.AntiAirSlash)
                return selectedMacro;

            return ComposeMacroAction(GetMovementMacro(selectedMacro), OffenseAction.HorizontalSlash);
        }

        private static bool ShouldEmitBasicAttackTap()
        {
            float now = Time.time;
            if (now <= attackTapPressUntil)
                return true;

            if (now < nextAttackTapStartAt)
                return false;

            attackTapPressUntil = now + BasicAttackTapSeconds;
            nextAttackTapStartAt = now + BasicAttackTapIntervalSeconds;
            return true;
        }

        public static bool IsInternallyConsistent(Action action)
        {
            if (action == null)
                return true;

            if (action.bind)
            {
                return action.move == MoveDirection.None &&
                    action.look == LookDirection.None &&
                    !action.jump &&
                    !action.attack &&
                    !action.dash &&
                    !action.needle &&
                    action.tool == ToolAction.None;
            }

            if (action.tool != ToolAction.None)
            {
                return !action.attack &&
                    !action.needle &&
                    !action.bind &&
                    (action.tool == ToolAction.Neutral ||
                        (action.tool == ToolAction.Up && action.look == LookDirection.Up) ||
                        (action.tool == ToolAction.Down && action.look == LookDirection.Down));
            }

            if (action.needle && (action.attack || action.bind || action.tool != ToolAction.None))
                return false;

            if (action.attack && action.dash && action.look != LookDirection.None)
                return false;

            return true;
        }

        /// <summary>
        /// Gets the key state for a given key name based on the translated raw action.
        /// Returns null if the key is not handled.
        /// </summary>
        public static bool? GetKeyState(string keyName, Action action)
        {
            if (action == null)
                return null;

            switch (keyName)
            {
                case "leftArrow":
                    return action.move == MoveDirection.Left;
                case "rightArrow":
                    return action.move == MoveDirection.Right;
                case "upArrow":
                    return action.tool != ToolAction.None
                        ? action.tool == ToolAction.Up
                        : action.look == LookDirection.Up;
                case "downArrow":
                    return action.tool != ToolAction.None
                        ? action.tool == ToolAction.Down
                        : action.look == LookDirection.Down;
                case "z":
                    return action.jump;
                case "x":
                    return action.attack;
                case "c":
                    return action.dash;
                case "s":
                    return action.needle;
                case "a":
                    return action.bind;
                case "f":
                    return action.tool != ToolAction.None;
            }

            return null;
        }

        public static MacroAction ClampMacroAction(int value)
        {
            if (value < 0 || value >= MacroActionCount)
            {
                RLManager.StaticLogger?.LogWarning($"[ActionManager] Unknown option action {value}, defaulting to HoldSafeSide");
                return MacroAction.HoldSafeSide;
            }

            return (MacroAction)value;
        }

        private static LowLevelAction ClampLowLevelAction(int value)
        {
            if (value < 0 || value >= LowLevelActionCount)
            {
                RLManager.StaticLogger?.LogWarning($"[ActionManager] Unknown low-level action {value}, defaulting to Idle");
                return LowLevelAction.Idle;
            }

            return (LowLevelAction)value;
        }

        private static LowLevelAction RawActionToLowLevelAction(Action action)
        {
            if (action == null)
                return LowLevelAction.Idle;

            if (action.bind)
                return LowLevelAction.BindHeal;

            if (action.tool != ToolAction.None)
            {
                if (action.tool == ToolAction.Up)
                    return LowLevelAction.ToolUp;
                if (action.tool == ToolAction.Down)
                    return LowLevelAction.ToolDown;
                if (action.jump && action.move == MoveDirection.Left)
                    return LowLevelAction.JumpLeftTool;
                if (action.jump && action.move == MoveDirection.Right)
                    return LowLevelAction.JumpRightTool;
                if (action.move == MoveDirection.Left)
                    return LowLevelAction.MoveLeftTool;
                if (action.move == MoveDirection.Right)
                    return LowLevelAction.MoveRightTool;
                return LowLevelAction.ToolNeutral;
            }

            if (action.needle)
                return LowLevelAction.Needle;

            if (action.attack)
            {
                if (action.dash && action.move == MoveDirection.Left)
                    return LowLevelAction.DashLeftAttack;
                if (action.dash && action.move == MoveDirection.Right)
                    return LowLevelAction.DashRightAttack;
                if (action.jump && action.move == MoveDirection.Left)
                    return LowLevelAction.JumpLeftAttack;
                if (action.jump && action.move == MoveDirection.Right)
                    return LowLevelAction.JumpRightAttack;
                if (action.jump)
                    return LowLevelAction.JumpAttack;
                if (action.look == LookDirection.Up)
                    return LowLevelAction.UpAttack;
                if (action.look == LookDirection.Down)
                    return LowLevelAction.DownAttack;
                if (action.move == MoveDirection.Left)
                    return LowLevelAction.MoveLeftAttack;
                if (action.move == MoveDirection.Right)
                    return LowLevelAction.MoveRightAttack;
                return LowLevelAction.Attack;
            }

            if (action.dash)
            {
                if (action.move == MoveDirection.Left)
                    return LowLevelAction.DashLeft;
                if (action.move == MoveDirection.Right)
                    return LowLevelAction.DashRight;
                return LowLevelAction.Dash;
            }

            if (action.jump)
            {
                if (action.move == MoveDirection.Left)
                    return LowLevelAction.JumpLeft;
                if (action.move == MoveDirection.Right)
                    return LowLevelAction.JumpRight;
                return LowLevelAction.Jump;
            }

            if (action.look == LookDirection.Up)
                return LowLevelAction.LookUp;
            if (action.look == LookDirection.Down)
                return LowLevelAction.LookDown;
            if (action.move == MoveDirection.Left)
                return LowLevelAction.MoveLeft;
            if (action.move == MoveDirection.Right)
                return LowLevelAction.MoveRight;

            return LowLevelAction.Idle;
        }

        public static string GetLowLevelActionName(LowLevelAction action)
        {
            return action.ToString();
        }

        private static Action CloneAction(Action source)
        {
            if (source == null)
                return new Action();

            return new Action
            {
                macro = source.macro,
                actionIndex = source.actionIndex,
                optionStarted = source.optionStarted,
                move = source.move,
                look = source.look,
                jump = source.jump,
                attack = source.attack,
                dash = source.dash,
                needle = source.needle,
                bind = source.bind,
                tool = source.tool
            };
        }

        private static MoveDirection GetApproachDirection(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            float dx = boss.transform.position.x - hero.transform.position.x;
            if (Mathf.Abs(dx) <= CollisionBufferRange)
                return GetDirectionAwayFromBoss(hero, boss);

            return dx >= 0f ? MoveDirection.Right : MoveDirection.Left;
        }

        private static MoveDirection GetKeepRangeDirection(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            float dx = boss.transform.position.x - hero.transform.position.x;
            float absX = Mathf.Abs(dx);
            if (absX > KeepRangeFar)
                return dx >= 0f ? MoveDirection.Right : MoveDirection.Left;
            if (absX < KeepRangeNear)
                return dx >= 0f ? MoveDirection.Left : MoveDirection.Right;
            if (!IsFacingBoss(hero, boss))
                return dx >= 0f ? MoveDirection.Right : MoveDirection.Left;

            return MoveDirection.None;
        }

        private static MoveDirection GetRetreatFromBossDodgeDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            return GetDirectionAwayFromBoss(hero, boss);
        }

        private static bool ShouldDashRetreat(HeroController hero, HealthManager boss, IBossEncounter encounter)
        {
            if (hero == null || boss == null || hero.cState.dashing)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            if (absY > RetreatDashVerticalTolerance)
                return false;

            if (IsBossDangerous(encounter, hero, boss) && absX <= RetreatDashDangerRange)
                return true;

            return absX <= RetreatDashPanicRange;
        }

        private static bool ShouldJumpRetreat(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            return !hero.cState.jumping && IsBossClose(hero, boss, CollisionBufferRange);
        }

        private static MoveDirection GetDodgeOverBossMoveDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            MoveDirection passDirection = EnsureDodgeOverBossPassDirection(hero, boss, encounter);
            if (IsIncomingGroundRushClose(hero, boss, encounter) && !HasDodgeOverBossHeight(hero, boss))
            {
                return MoveDirection.None;
            }

            if (!HasDodgeOverBossHeight(hero, boss))
                return passDirection;

            if (!HasDodgeOverBossPassedWithClearance(hero, boss, passDirection))
                return passDirection;

            if (IsBossDangerous(encounter, hero, boss) || IsBossMovingTowardHero(hero, boss))
            {
                return passDirection;
            }

            return MoveDirection.None;
        }

        private static MoveDirection EnsureDodgeOverBossPassDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (dodgeOverBossPassDirection != MoveDirection.None)
                return dodgeOverBossPassDirection;

            MoveDirection towardBoss = GetDirectionToBoss(hero, boss);
            MoveDirection oppositeBossVelocity = GetDirectionOppositeBossVelocity(boss);
            if (oppositeBossVelocity != MoveDirection.None && oppositeBossVelocity == towardBoss)
            {
                dodgeOverBossPassDirection = oppositeBossVelocity;
            }
            else
            {
                dodgeOverBossPassDirection = towardBoss != MoveDirection.None
                    ? towardBoss
                    : GetDirectionToCenter(hero, encounter);
            }

            return dodgeOverBossPassDirection;
        }

        private static bool HasDodgeOverBossPassedWithClearance(
            HeroController hero,
            HealthManager boss,
            MoveDirection passDirection)
        {
            if (hero == null || boss == null)
                return false;

            float heroX = hero.transform.position.x;
            float bossX = boss.transform.position.x;
            if (passDirection == MoveDirection.Left)
                return heroX <= bossX - DodgeOverBossPassClearance;
            if (passDirection == MoveDirection.Right)
                return heroX >= bossX + DodgeOverBossPassClearance;

            return false;
        }

        private static bool ShouldAirDashThrough(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter,
            bool requireCompletedHighJump = false)
        {
            if (hero == null || boss == null || hero.cState.dashing)
                return false;

            if (!hero.cState.jumping)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            if (requireCompletedHighJump && ShouldKeepHoldingDodgeOverBossJump(hero, boss))
                return false;

            if (!HasDodgeOverBossHeight(hero, boss))
                return false;

            if (absX <= 1.7f)
                return true;

            if (absX <= DodgeAirDashHorizontalRange &&
                absY <= DodgeAirDashVerticalTolerance &&
                (IsBossDangerous(encounter, hero, boss) || IsBossMovingTowardHero(hero, boss)))
            {
                return true;
            }

            return false;
        }

        private static bool ShouldKeepHoldingDodgeOverBossJump(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            float sustainedElapsed = GetMovementMacro(defensiveSustainedMacro) == MacroAction.DodgeOverBossHighJump &&
                defensiveOptionStartedAt > 0f
                ? Time.fixedTime - defensiveOptionStartedAt
                : 0f;

            if (sustainedElapsed < DodgeOverBossMinJumpHoldSeconds)
                return true;

            return false;
        }

        private static void ApplyDodgeOverBossJumpBudget(
            Action action,
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (action == null || hero == null || boss == null)
                return;

            EnsureDodgeOverBossJumpCycleStarted();

            if (action.dash)
            {
                action.jump = false;
                return;
            }

            bool airborne = IsHeroAirborne(hero);
            if (!airborne)
            {
                dodgeOverBossWasAirborne = false;
                dodgeOverBossAirJumpUsed = false;
                dodgeOverBossAirJumpHoldUntil = -1f;
                action.jump = true;
                return;
            }

            dodgeOverBossWasAirborne = true;
            float elapsed = GetDodgeOverBossJumpElapsed();
            if (elapsed < DodgeOverBossMinJumpHoldSeconds)
            {
                action.jump = true;
                return;
            }

            if (elapsed < DodgeOverBossMinJumpHoldSeconds + DodgeOverBossJumpReleaseGapSeconds)
            {
                action.jump = false;
                return;
            }

            if (Time.fixedTime <= dodgeOverBossAirJumpHoldUntil)
            {
                action.jump = true;
                return;
            }

            if (!dodgeOverBossAirJumpUsed && ShouldUseDodgeOverBossAirJump(hero, boss, encounter))
            {
                dodgeOverBossAirJumpUsed = true;
                dodgeOverBossAirJumpHoldUntil = Time.fixedTime + DodgeOverBossAirJumpHoldSeconds;
                action.jump = true;
                return;
            }

            action.jump = false;
        }

        private static void EnsureDodgeOverBossJumpCycleStarted()
        {
            if (dodgeOverBossJumpCycleStartedAt < 0f)
            {
                StartDodgeOverBossJumpCycle(Time.fixedTime);
            }
        }

        private static void StartDodgeOverBossJumpCycle(float startedAt)
        {
            dodgeOverBossJumpCycleStartedAt = startedAt;
            dodgeOverBossAirJumpHoldUntil = -1f;
            dodgeOverBossPassDirection = MoveDirection.None;
            dodgeOverBossWasAirborne = false;
            dodgeOverBossAirJumpUsed = false;
        }

        private static void ResetDodgeOverBossJumpCycle()
        {
            dodgeOverBossJumpCycleStartedAt = -1f;
            dodgeOverBossAirJumpHoldUntil = -1f;
            dodgeOverBossPassDirection = MoveDirection.None;
            dodgeOverBossWasAirborne = false;
            dodgeOverBossAirJumpUsed = false;
        }

        private static float GetDodgeOverBossJumpElapsed()
        {
            float startedAt = dodgeOverBossJumpCycleStartedAt >= 0f
                ? dodgeOverBossJumpCycleStartedAt
                : defensiveOptionStartedAt;
            if (startedAt < 0f)
                return 0f;

            return Mathf.Max(0f, Time.fixedTime - startedAt);
        }

        private static bool IsHeroAirborne(HeroController hero)
        {
            if (hero == null)
                return false;

            if (hero.cState.jumping)
                return true;

            Rigidbody2D rb = hero.GetComponent<Rigidbody2D>();
            return rb != null && Mathf.Abs(rb.linearVelocity.y) > 0.2f;
        }

        private static bool ShouldUseDodgeOverBossAirJump(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (!dodgeOverBossWasAirborne)
                return false;

            if (!HasDodgeOverBossHeight(hero, boss))
                return true;

            return IsBossDangerous(encounter, hero, boss) && IsBossClose(hero, boss, JumpDangerRange);
        }

        private static bool HasDodgeOverBossHeight(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            return hero.transform.position.y - boss.transform.position.y >= DodgeOverBossMinHeroAboveBoss;
        }

        private static MoveDirection GetDodgeThroughDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            if (dodgeOverBossPassDirection != MoveDirection.None)
                return dodgeOverBossPassDirection;

            if (IsIncomingGroundRushClose(hero, boss, encounter) ||
                IsBossMovingTowardHero(hero, boss) ||
                IsBossDangerous(encounter, hero, boss))
            {
                MoveDirection oppositeBossVelocity = GetDirectionOppositeBossVelocity(boss);
                if (oppositeBossVelocity != MoveDirection.None)
                    return oppositeBossVelocity;
            }

            return GetDirectionToBoss(hero, boss);
        }

        private static MoveDirection GetCrossUnderAirBossDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            float dx = boss.transform.position.x - hero.transform.position.x;
            float dy = boss.transform.position.y - hero.transform.position.y;
            if (dy < 1.0f)
                return GetHoldSafeSideDirection(hero, boss, encounter);

            if (Mathf.Abs(dx) > 0.6f)
                return dx >= 0f ? MoveDirection.Right : MoveDirection.Left;

            return GetDirectionToCenter(hero, encounter);
        }

        private static bool ShouldDashCrossUnder(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || hero.cState.dashing)
                return false;

            float dx = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float dy = boss.transform.position.y - hero.transform.position.y;
            return dy > 2.0f && dx > 2.5f;
        }

        private static bool ShouldJumpOverBossToReachCenter(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null || encounter == null)
                return false;

            MoveDirection center = GetDirectionToCenter(hero, encounter);
            if (center == MoveDirection.None || center != GetDirectionToBoss(hero, boss))
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            if (absY > 1.7f)
                return false;

            if (IsBossDangerous(encounter, hero, boss) && absX <= 2.8f)
                return true;

            return absX <= CollisionBufferRange + 0.7f;
        }

        private static MoveDirection GetSafeCenterDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || encounter == null)
                return MoveDirection.None;

            MoveDirection center = GetDirectionToCenter(hero, encounter);
            if (center == MoveDirection.None || boss == null)
                return center;

            if (IsBossDangerous(encounter, hero, boss) &&
                IsCenterPathBlockedByBoss(hero, boss, center, 2.5f))
            {
                return GetDirectionAwayFromBoss(hero, boss);
            }

            if (IsCenterPathBlockedByBoss(hero, boss, center, CollisionBufferRange + 0.45f))
            {
                return MoveDirection.None;
            }

            return center;
        }

        private static bool IsCenterPathBlockedByBoss(
            HeroController hero,
            HealthManager boss,
            MoveDirection centerDirection,
            float horizontalRange)
        {
            if (hero == null || boss == null || centerDirection == MoveDirection.None)
                return false;

            if (centerDirection != GetDirectionToBoss(hero, boss))
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            return absX <= horizontalRange && absY <= 1.7f;
        }

        private static bool ShouldDashForPunish(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null || hero.cState.dashing)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            return absX > KeepRangeFar + 1.0f;
        }

        private static MoveDirection GetHoldSafeSideDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            float dx = boss.transform.position.x - hero.transform.position.x;
            float absX = Mathf.Abs(dx);
            if (absX < KeepRangeNear + 0.4f)
                return GetDirectionAwayFromBoss(hero, boss);
            if (absX > KeepRangeFar + 1.0f)
                return GetDirectionToBoss(hero, boss);

            if (!IsFacingBoss(hero, boss))
                return GetDirectionToBoss(hero, boss);

            MoveDirection center = GetDirectionToCenter(hero, encounter);
            if (center != MoveDirection.None && IsFarFromCenter(hero, encounter))
                return center;

            return MoveDirection.None;
        }

        private static MoveDirection GetAntiAirPositionDirection(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            float dx = boss.transform.position.x - hero.transform.position.x;
            if (Mathf.Abs(dx) > VerticalAttackHorizontalToleranceFallback)
                return dx >= 0f ? MoveDirection.Right : MoveDirection.Left;

            return MoveDirection.None;
        }

        private static ToolAction GetRangedToolAction(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return ToolAction.Down;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            return absX > 5.0f ? ToolAction.Down : ToolAction.Up;
        }

        private static MoveDirection GetHorizontalSlashDirection(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            float dx = boss.transform.position.x - hero.transform.position.x;
            float absX = Mathf.Abs(dx);
            MoveDirection towardBoss = dx >= 0f ? MoveDirection.Right : MoveDirection.Left;

            if (absX < SlashRetreatRange)
                return Opposite(towardBoss);

            if (!IsFacingBoss(hero, boss) || absX > SlashStepTowardRange)
                return towardBoss;

            return MoveDirection.None;
        }

        private static MoveDirection GetJumpDirection(HeroController hero, HealthManager boss, IBossEncounter encounter)
        {
            if (hero == null)
                return MoveDirection.None;

            MoveDirection centerDirection = GetDirectionToCenter(hero, encounter);
            if (centerDirection != MoveDirection.None && IsFarFromCenter(hero, encounter))
                return centerDirection;

            if (boss != null)
            {
                if (IsBossDangerous(encounter, hero, boss) && IsBossClose(hero, boss, JumpDangerRange))
                    return GetDirectionAwayFromBoss(hero, boss);

                if (IsBossClose(hero, boss, CollisionBufferRange))
                    return GetDirectionAwayFromBoss(hero, boss);
            }

            return MoveDirection.None;
        }

        private static MoveDirection GetDashThroughDirection(HeroController hero, HealthManager boss, IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            MoveDirection centerDirection = GetDirectionToCenter(hero, encounter);
            if (centerDirection != MoveDirection.None &&
                IsFarFromCenter(hero, encounter) &&
                IsBossDangerous(encounter, hero, boss))
            {
                return centerDirection;
            }

            if (IsBossDangerous(encounter, hero, boss) && IsBossClose(hero, boss, DashThroughDangerRange))
                return GetDirectionAwayFromBoss(hero, boss);

            return GetDirectionToBoss(hero, boss);
        }

        private static MoveDirection GetDirectionToBoss(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return MoveDirection.None;

            return boss.transform.position.x >= hero.transform.position.x
                ? MoveDirection.Right
                : MoveDirection.Left;
        }

        private static bool IsBossClose(HeroController hero, HealthManager boss, float maxDistance)
        {
            if (hero == null || boss == null)
                return false;

            return Vector2.Distance(hero.transform.position, boss.transform.position) <= maxDistance;
        }

        private static bool IsIncomingGroundRushClose(HeroController hero, HealthManager boss, IBossEncounter encounter)
        {
            if (hero == null || boss == null)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            return absX <= DodgeIncomingHoldHorizontalRange &&
                absY <= DodgeIncomingHoldVerticalTolerance &&
                IsBossDangerous(encounter, hero, boss) &&
                IsBossMovingTowardHero(hero, boss);
        }

        private static bool IsBossMovingTowardHero(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            Rigidbody2D bossRb = boss.GetComponent<Rigidbody2D>();
            if (bossRb == null)
                return false;

            float bossVelX = bossRb.linearVelocity.x;
            float relX = hero.transform.position.x - boss.transform.position.x;
            if (Mathf.Abs(bossVelX) < DodgeIncomingMinBossSpeed)
                return false;

            return Mathf.Sign(bossVelX) == Mathf.Sign(relX);
        }

        private static MoveDirection GetDirectionOppositeBossVelocity(HealthManager boss)
        {
            if (boss == null)
                return MoveDirection.None;

            Rigidbody2D bossRb = boss.GetComponent<Rigidbody2D>();
            if (bossRb == null)
                return MoveDirection.None;

            float bossVelX = bossRb.linearVelocity.x;
            if (Mathf.Abs(bossVelX) < 0.1f)
                return MoveDirection.None;

            return bossVelX > 0f ? MoveDirection.Left : MoveDirection.Right;
        }

        private static bool IsWithinKeepRange(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return false;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            return absX >= KeepRangeNear && absX <= KeepRangeFar;
        }

        private static bool IsBossDangerous(IBossEncounter encounter, HeroController hero, HealthManager boss)
        {
            if (encounter == null)
                return false;

            try
            {
                return encounter.IsBossDangerous(hero, boss);
            }
            catch (Exception e)
            {
                RLManager.StaticLogger?.LogWarning($"[ActionManager] Failed to query boss danger state: {e.Message}");
                return false;
            }
        }

        private static bool IsDefensiveCommitMacro(MacroAction macro)
        {
            MacroAction movementMacro = GetMovementMacro(macro);
            OffenseAction offense = GetOffenseAction(macro);
            if (offense == OffenseAction.RangedTool)
                return false;

            return movementMacro == MacroAction.DodgeOverBossHighJump ||
                movementMacro == MacroAction.CrossUnderAirBoss ||
                movementMacro == MacroAction.RetreatFromBossDodge ||
                movementMacro == MacroAction.MoveToCenterSafely;
        }

        private static float GetSustainedOptionSeconds(MacroAction macro)
        {
            float baseSeconds;
            switch (GetMovementMacro(macro))
            {
                case MacroAction.DodgeOverBossHighJump:
                    baseSeconds = DodgeOverBossHighJumpOptionSeconds;
                    break;
                case MacroAction.CrossUnderAirBoss:
                    baseSeconds = CrossUnderAirBossOptionSeconds;
                    break;
                case MacroAction.RetreatFromBossDodge:
                    baseSeconds = RetreatFromBossDodgeOptionSeconds;
                    break;
                case MacroAction.MoveToCenterSafely:
                    baseSeconds = MoveToCenterSafelyOptionSeconds;
                    break;
                default:
                    return 0f;
            }

            return Mathf.Max(0.05f, baseSeconds + UnityEngine.Random.Range(-OptionDurationJitterSeconds, OptionDurationJitterSeconds));
        }

        private static int GetDefensivePriority(MacroAction macro)
        {
            switch (GetMovementMacro(macro))
            {
                case MacroAction.DodgeOverBossHighJump:
                    return 4;
                case MacroAction.CrossUnderAirBoss:
                    return 3;
                case MacroAction.MoveToCenterSafely:
                    return 2;
                case MacroAction.RetreatFromBossDodge:
                    return 1;
                default:
                    return 0;
            }
        }

        private static bool TryGetDefensiveInterrupt(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter,
            MacroAction currentMacro,
            out MacroAction interruptMacro)
        {
            interruptMacro = currentMacro;
            if (hero == null || boss == null || encounter == null)
                return false;

            int[] mask = encounter.GetActionMask(hero, boss);
            MacroAction teacherMacro = encounter.GetTeacherAction(hero, boss, mask);
            if (!IsDefensiveCommitMacro(teacherMacro))
                return false;

            bool strongerDefense = GetDefensivePriority(teacherMacro) > GetDefensivePriority(currentMacro);
            if (!strongerDefense)
                return false;

            interruptMacro = teacherMacro;
            return true;
        }

        private static bool ShouldCancelDefensiveOption(
            HeroController hero,
            HealthManager boss,
            IBossEncounter encounter,
            MacroAction macro)
        {
            if (hero == null || encounter == null)
                return true;

            MacroAction movementMacro = GetMovementMacro(macro);
            if (movementMacro == MacroAction.DodgeOverBossHighJump &&
                defensiveOptionStartedAt > 0f &&
                Time.fixedTime - defensiveOptionStartedAt < DodgeOverBossMinJumpHoldSeconds)
            {
                return false;
            }

            if (hero.cState.jumping || IsBossDangerous(encounter, hero, boss))
                return false;

            return movementMacro == MacroAction.DodgeOverBossHighJump ||
                movementMacro == MacroAction.CrossUnderAirBoss;
        }

        private static MoveDirection GetDirectionAwayFromBoss(HeroController hero, HealthManager boss)
        {
            return Opposite(GetDirectionToBoss(hero, boss));
        }

        private static MoveDirection GetDirectionToCenter(HeroController hero, IBossEncounter encounter)
        {
            if (hero == null || encounter == null)
                return MoveDirection.None;

            float centerX = encounter.GetPreferredCenterX();
            float dx = centerX - hero.transform.position.x;
            if (Mathf.Abs(dx) <= CenterDeadzone)
                return MoveDirection.None;

            return dx >= 0f ? MoveDirection.Right : MoveDirection.Left;
        }

        private static bool IsFarFromCenter(HeroController hero, IBossEncounter encounter)
        {
            if (hero == null || encounter == null)
                return false;

            return Mathf.Abs(encounter.GetPreferredCenterX() - hero.transform.position.x) > 3f;
        }

        private static void ClampMovementToSafeArea(Action action, HeroController hero, IBossEncounter encounter)
        {
            if (action == null || encounter == null)
                return;

            if (action.bind)
                return;

            try
            {
                MoveDirection requestedMove = action.move;
                action.move = encounter.ClampMovementToSafeArea(hero, action.move);
                if (requestedMove != action.move)
                {
                    if (requestedMove != MoveDirection.None)
                    {
                        action.dash = false;
                    }

                    if (requestedMove != MoveDirection.None &&
                        GetMovementMacro(action.macro) != MacroAction.DodgeOverBossHighJump)
                    {
                        action.jump = false;
                    }
                }
            }
            catch (Exception e)
            {
                RLManager.StaticLogger?.LogWarning($"[ActionManager] Failed to clamp movement to safe area: {e.Message}");
            }
        }

        private static MoveDirection Opposite(MoveDirection direction)
        {
            if (direction == MoveDirection.Left)
                return MoveDirection.Right;
            if (direction == MoveDirection.Right)
                return MoveDirection.Left;
            return MoveDirection.None;
        }

        private static void FaceBossIfNeeded(Action action, HeroController hero, HealthManager boss)
        {
            if (action == null || hero == null || boss == null || IsFacingBoss(hero, boss))
                return;

            action.move = GetDirectionToBoss(hero, boss);
        }

        private static void FaceBossForHorizontalSlash(Action action, HeroController hero, HealthManager boss)
        {
            if (action == null || hero == null || boss == null || !action.attack || action.look != LookDirection.None)
                return;

            float absX = Mathf.Abs(boss.transform.position.x - hero.transform.position.x);
            float absY = Mathf.Abs(boss.transform.position.y - hero.transform.position.y);
            if (absX <= 4.0f && absY <= 1.8f)
            {
                action.move = GetDirectionToBoss(hero, boss);
                action.dash = false;
            }
            else
            {
                FaceBossIfNeeded(action, hero, boss);
            }
        }

        private static void FaceBossIfStationary(Action action, HeroController hero, HealthManager boss)
        {
            if (action == null || action.move != MoveDirection.None)
                return;

            FaceBossIfNeeded(action, hero, boss);
        }

        private static bool IsFacingBoss(HeroController hero, HealthManager boss)
        {
            if (hero == null || boss == null)
                return true;

            bool bossIsRight = boss.transform.position.x >= hero.transform.position.x;
            return bossIsRight ? hero.cState.facingRight : !hero.cState.facingRight;
        }
    }

    // Harmony patch for the new Unity Input System
    // Patches ButtonControl.isPressed to intercept key presses
    [HarmonyPatch(typeof(ButtonControl), "get_isPressed")]
    public static class ButtonControlPatch
    {
        public static bool Prefix(ButtonControl __instance, ref bool __result)
        {
            if (!RLManager.isAgentControlEnabled)
                return true;

            var action = RLManager.currentAction;
            if (action == null)
                return true;

            string keyName = __instance.name;

            bool? keyState = ActionManager.GetKeyState(keyName, action);
            if (keyState.HasValue)
            {
                __result = keyState.Value;
                return false;
            }

            return true;
        }
    }

    /* Debug still uses old input system, so we use the patch below
    keeping this just in case it gets an update later

    [HarmonyPatch(typeof(ButtonControl), "get_wasPressedThisFrame")]
    public static class ButtonControlWasPressedPatch
    {
        public static bool Prefix(ButtonControl __instance, ref bool __result)
        {
            if (!RLManager.isAgentControlEnabled)
                return true;

            string keyName = __instance.name;

            // Handle F5 key down for reset functionality
            if (keyName == "f5")
            {
                if (RLManager.simulateF5Press)
                {
                    __result = true;
                    return false;
                }
            }

            return true;
        }
    }
    */

    // Patch over the legacy input system because that's what the
    // Debug Mod uses
    [HarmonyPatch(typeof(Input), "GetKeyDown", typeof(KeyCode))]
    public static class GetKeyDownPatch
    {
        public static bool Prefix(KeyCode key, ref bool __result)
        {
            if (!RLManager.isAgentControlEnabled)
                return true;

            if (key == KeyCode.F5)
            {
                if (RLManager.IsSimulatingF5Press())
                {
                    __result = true;
                    return false;
                }
            }
            
            return true;
        }
    }

    [HarmonyPatch(typeof(Input), "GetKey", typeof(KeyCode))]
    public static class GetKeyPatch
    {
        public static bool Prefix(KeyCode key, ref bool __result)
        {
            if (!RLManager.isAgentControlEnabled)
                return true;

            if (key == KeyCode.F5 && RLManager.IsSimulatingF5Press())
            {
                __result = true;
                return false;
            }

            return true;
        }
    }
}
