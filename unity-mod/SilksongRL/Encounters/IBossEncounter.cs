using System;
using UnityEngine;

namespace SilksongRL
{
    [Serializable]
    public class RewardComponents
    {
        public float damageReward;
        public float healReward;
        public float attackReward;
        public float baseSurvivalReward;
        public float dodgeReward;
        public float resourceReward;
        public float positionReward;
        public float phaseReward;
        public float heroDamagePenalty;

        public RewardComponents Clone()
        {
            return new RewardComponents
            {
                damageReward = damageReward,
                healReward = healReward,
                attackReward = attackReward,
                baseSurvivalReward = baseSurvivalReward,
                dodgeReward = dodgeReward,
                resourceReward = resourceReward,
                positionReward = positionReward,
                phaseReward = phaseReward,
                heroDamagePenalty = heroDamagePenalty
            };
        }

        public float[] ToArray()
        {
            return new[]
            {
                damageReward,
                healReward,
                attackReward,
                baseSurvivalReward,
                dodgeReward,
                resourceReward,
                positionReward,
                phaseReward,
                heroDamagePenalty
            };
        }
    }

    /// <summary>
    /// Observation types for different encounters.
    /// Tells Python how to process the observation array.
    /// </summary>
    public enum ObservationType
    {
        // Vector only: flat array of state values
        //目前的向量也许太过简单
        Vector,

        // Hybrid: [vector_obs | visual_obs] - split and process separately
        //综合考虑性能，也许截图识别并不是一个较好的选择
        //当然，如果boss战的场地信息无法通过harmonypatch得到，
        //则视觉识别是有必要的
        //再次考虑，假设boss会召唤小怪呢？能用harmonypatch得到吗？
        Hybrid
    }

    /// <summary>
    /// Interface for boss encounter configurations.
    /// Each boss encounter implements this to define its specific behavior,
    /// observation space, action space, and reset mechanics.
    /// </summary>
    public interface IBossEncounter
    {
        /// <summary>
        /// Gets the human-readable name of this encounter.
        /// </summary>
        string GetEncounterName();

        /// <summary>
        /// Gets the observation type for this encounter.
        /// Vector = flat state values only.
        /// Hybrid = [vector_obs | visual_obs] for CNN processing.
        /// </summary>
        ObservationType GetObservationType();

        /// <summary>
        /// Gets the size of the vector portion of observations.
        /// For Vector type, this equals GetObservationSize().
        /// For Hybrid type, this is just the vector part (before visual data).
        /// </summary>
        int GetVectorObservationSize();

        /// <summary>
        /// Gets the visual observation dimensions.
        /// Returns (0, 0) for vector-only observations.
        /// </summary>
        (int width, int height) GetVisualObservationSize();

        /// <summary>
        /// Checks if the given HealthManager matches this encounter.
        /// </summary>
        bool IsEncounterMatch(HealthManager hm);

        /// <summary>
        /// Extracts observations from the current game state.
        /// This allows each encounter to define its own observation space
        /// (e.g., base observations, projectiles, summons, environmental hazards).
        /// </summary>
        /// 很重要的函数，决定提取内容，待优化的点：更多参数，返回更多内容
        float[] ExtractObservationArray(HeroController hero, HealthManager boss);

        /// <summary>
        /// Returns a flattened invalid-action mask matching ActionManager.GetActionSpaceShape().
        /// 1 means the action value is valid, 0 means it should be masked out.
        /// </summary>
        int[] GetActionMask(HeroController hero, HealthManager boss);

        /// <summary>
        /// Returns the preferred horizontal center for macro movement options.
        /// Encounters with a safe platform should return that platform's center.
        /// </summary>
        float GetPreferredCenterX();

        /// <summary>
        /// Returns whether the boss is currently in a dangerous/active attack state.
        /// Used by macro actions to choose safer low-level movement.
        /// </summary>
        bool IsBossDangerous(HeroController hero, HealthManager boss);

        /// <summary>
        /// Returns a rule-based teacher macro action for the current state.
        /// This is used as an online auxiliary target for the policy.
        /// </summary>
        MacroAction GetTeacherAction(HeroController hero, HealthManager boss, int[] actionMask);

        /// <summary>
        /// Returns whether normal movement macros should be converted into short basic-attack taps.
        /// </summary>
        bool ShouldForceBasicAttack();

        /// <summary>
        /// Clamps a horizontal movement direction so sustained actions cannot walk off a safe arena platform.
        /// </summary>
        MoveDirection ClampMovementToSafeArea(HeroController hero, MoveDirection requestedDirection);

        /// <summary>
        /// Returns the size of the observation array for this encounter.
        /// Must match the length of the array returned by ExtractObservationArray().
        /// </summary>
        int GetObservationSize();

        /// <summary>
        /// Clears any temporal observation state such as frame stacking.
        /// Call this when an episode or scene reset starts.
        /// </summary>
        void ResetObservationHistory();

        /// <summary>
        /// Calculates the reward for the current transition.
        /// Each encounter can define its own reward function.
        /// </summary>
        /// 待优化：不一定每个boss专门定义一个专属reword，而是期望训练出更泛化的打boss策略
        float CalculateReward(float[] previousObservations, float[] currentObservations, Action previousAction, int whoDied);

        RewardComponents GetLastRewardComponents();

        /// <summary>
        /// Creates a terminal next observation when the live boss or hero object is already gone.
        /// </summary>
        float[] CreateTerminalObservation(float[] previousObservations, int whoDied);

        /// <summary>
        /// Checks if the hero is stuck.
        /// Conditions are arena/boss specific.
        /// Might not be required for all encounters.
        /// </summary>
        bool IsHeroStuck(HeroController hero);

        /// <summary>
        /// Gets the screen capture instance for hybrid observations.
        /// Returns null for vector-only encounters.
        /// </summary>
        ScreenCapture GetScreenCapture();

        /// <summary>
        /// Gets the maximum HP of the boss.
        /// </summary>
        float GetMaxHP();


        // NOTE:
        // The following three methods are not currently used.
        // They were needed because with the previous resetting mechanism
        // the fight would not trigger immediately. This may still happen
        // in certain encounters where we cannot reset straight into the fight.
        // but would first need to move a bit to trigger it so keeping them
        // just in case.

        /*
        /// <summary>
        /// Checks if the boss is in a dormant/inactive state.
        /// Used during reset sequences to determine when the fight can resume.
        /// </summary>
        bool IsBossDormant(HealthManager boss);

        /// <summary>
        /// Returns the action the hero should take during reset to initiate the fight.
        /// </summary>
        Action GetResetAction(HeroController hero, HealthManager boss);

        /// <summary>
        /// Checks if the reset sequence is complete and training can resume.
        /// </summary>
        bool IsResetComplete(HeroController hero, HealthManager boss);
        */

        //以上三个函数可能会尝试启用，假设我希望进行这样的训练：
        //对于多个用于训练的boss，不论输赢，总是一个个循环挑战过去
        //那么单靠DebugMod savestate / Quickslot可能不行，
        //需要以上三个函数来在游戏内调整挑战的boss
    }
}

