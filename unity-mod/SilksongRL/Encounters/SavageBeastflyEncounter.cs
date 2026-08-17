namespace SilksongRL
{
    public class SavageBeastflyEncounter : BossEncounterBase
    {
        protected override string BossName => "Bone Flyer Giant";
        protected override float MinPosX => 38f;
        protected override float MaxPosX => 65f;
        protected override float MinPosY => 34f;
        protected override float MaxPosY => 44f;
        protected override float MaxBossHP => 550f;
        protected override float MaxBossVelocity => 30f;
        protected override float? LowYWarningThreshold => null;

        public override bool IsHeroStuck(HeroController hero)
        {
            return false;
        }
    }
}
