using System;
using System.Text;
using UnityEngine;

namespace SilksongRL
{
    public class HitboxDebugLogger : MonoBehaviour
    {
        private const float LogIntervalSeconds = 0.5f;
        private const float AttackLogIntervalSeconds = 0.05f;
        private const float NearbyDistance = 16f;
        private const int MaxLoggedColliders = 80;
        private const int MaxAttackLoggedColliders = 32;

        private float nextLogTime;
        private float nextAttackLogTime;
        private string lastAttackSignature = "";

        public bool Enabled { get; set; }

        public void Tick(HeroController hero, HealthManager boss)
        {
            if (!Enabled)
                return;

            if (Time.unscaledTime >= nextAttackLogTime)
            {
                nextAttackLogTime = Time.unscaledTime + AttackLogIntervalSeconds;
                LogAttackColliders(hero, boss);
            }

            if (Time.unscaledTime >= nextLogTime)
            {
                nextLogTime = Time.unscaledTime + LogIntervalSeconds;
                LogColliders(hero, boss);
            }
        }

        private void LogAttackColliders(HeroController hero, HealthManager boss)
        {
            Collider2D[] colliders = FindObjectsOfType<Collider2D>();
            StringBuilder builder = new StringBuilder(4096);
            StringBuilder signature = new StringBuilder(1024);
            int logged = 0;

            builder.Append("[HitboxDebug/Attack] active attack colliders");
            if (hero != null)
            {
                Vector3 pos = hero.transform.position;
                builder.AppendFormat(" hero=({0:F2},{1:F2})", pos.x, pos.y);
            }
            if (boss != null)
            {
                Vector3 pos = boss.transform.position;
                builder.AppendFormat(" boss=({0:F2},{1:F2})", pos.x, pos.y);
            }

            foreach (Collider2D collider in colliders)
            {
                if (!IsAttackRelevant(collider, hero, boss))
                    continue;

                Bounds bounds = collider.bounds;
                string path = GetPath(collider.transform);
                signature.Append(path)
                    .Append('|').Append(collider.gameObject.layer)
                    .Append('|').Append(Mathf.RoundToInt(bounds.center.x * 10f))
                    .Append('|').Append(Mathf.RoundToInt(bounds.center.y * 10f))
                    .Append('|').Append(Mathf.RoundToInt(bounds.size.x * 10f))
                    .Append('|').Append(Mathf.RoundToInt(bounds.size.y * 10f))
                    .Append(';');

                if (logged >= MaxAttackLoggedColliders)
                    continue;

                builder.AppendLine();
                builder.AppendFormat(
                    "  #{0} {1} path='{2}' layer={3} trigger={4} center=({5:F2},{6:F2}) size=({7:F2},{8:F2}) tag={9}",
                    logged + 1,
                    collider.GetType().Name,
                    path,
                    collider.gameObject.layer,
                    collider.isTrigger,
                    bounds.center.x,
                    bounds.center.y,
                    bounds.size.x,
                    bounds.size.y,
                    Classify(collider, boss));
                logged++;
            }

            string currentSignature = signature.ToString();
            if (logged == 0)
            {
                lastAttackSignature = "";
                return;
            }

            if (currentSignature == lastAttackSignature)
                return;

            lastAttackSignature = currentSignature;
            RLManager.StaticLogger?.LogInfo(builder.ToString());
        }

        private void LogColliders(HeroController hero, HealthManager boss)
        {
            Collider2D[] colliders = FindObjectsOfType<Collider2D>();
            StringBuilder builder = new StringBuilder(8192);
            int logged = 0;
            int candidates = 0;

            builder.Append("[HitboxDebug] active relevant colliders");
            if (hero != null)
            {
                Vector3 pos = hero.transform.position;
                builder.AppendFormat(" hero=({0:F2},{1:F2})", pos.x, pos.y);
            }
            if (boss != null)
            {
                Vector3 pos = boss.transform.position;
                builder.AppendFormat(" boss=({0:F2},{1:F2})", pos.x, pos.y);
            }

            foreach (Collider2D collider in colliders)
            {
                if (!IsRelevant(collider, hero, boss))
                    continue;

                candidates++;
                if (logged >= MaxLoggedColliders)
                    continue;

                Bounds bounds = collider.bounds;
                builder.AppendLine();
                builder.AppendFormat(
                    "  #{0} {1} path='{2}' layer={3} trigger={4} enabled={5} center=({6:F2},{7:F2}) size=({8:F2},{9:F2}) tag={10}",
                    logged + 1,
                    collider.GetType().Name,
                    GetPath(collider.transform),
                    collider.gameObject.layer,
                    collider.isTrigger,
                    collider.enabled,
                    bounds.center.x,
                    bounds.center.y,
                    bounds.size.x,
                    bounds.size.y,
                    Classify(collider, boss));
                logged++;
            }

            if (candidates > logged)
            {
                builder.AppendLine();
                builder.AppendFormat("  ... {0} more relevant colliders omitted", candidates - logged);
            }

            RLManager.StaticLogger?.LogInfo(builder.ToString());
        }

        private bool IsRelevant(Collider2D collider, HeroController hero, HealthManager boss)
        {
            if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy)
                return false;

            if (hero != null && collider.GetComponentInParent<HeroController>() == hero)
                return true;

            if (boss != null && collider.GetComponentInParent<HealthManager>() == boss)
                return true;

            if (collider.GetComponent<DamageHero>() != null || collider.GetComponentInParent<DamageHero>() != null)
                return true;

            if (collider.GetComponent<HealthManager>() != null || collider.GetComponentInParent<HealthManager>() != null)
                return true;

            string name = collider.gameObject.name.ToLowerInvariant();
            if (name.Contains("hit") ||
                name.Contains("slash") ||
                name.Contains("damager") ||
                name.Contains("projectile") ||
                name.Contains("needle") ||
                name.Contains("lace_circle"))
            {
                return true;
            }

            if (hero != null)
            {
                Vector2 heroPos = hero.transform.position;
                Vector2 colliderPos = collider.bounds.center;
                return Vector2.Distance(heroPos, colliderPos) <= NearbyDistance;
            }

            return false;
        }

        private bool IsAttackRelevant(Collider2D collider, HeroController hero, HealthManager boss)
        {
            if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy)
                return false;

            string path = GetPath(collider.transform).ToLowerInvariant();
            if (path.Contains("nothwip range") ||
                path.Contains("evade range") ||
                path.Contains("above range") ||
                path.Contains("wall range"))
            {
                return false;
            }

            if (IsStaticStageHazardPath(path))
                return false;

            if (hero != null &&
                collider.GetComponentInParent<HeroController>() == hero &&
                path.Contains("/attacks/"))
            {
                return true;
            }

            if (IsStandaloneBossAttackPath(collider, path))
                return true;

            if (boss != null && collider.GetComponentInParent<HealthManager>() == boss)
            {
                return collider.gameObject.layer == 22 ||
                    path.Contains("hit") ||
                    path.Contains("slash") ||
                    path.Contains("multihit") ||
                    path.Contains("bodycatcher") ||
                    path.Contains("damager") ||
                    path.Contains("projectile");
            }

            return collider.GetComponent<DamageHero>() != null ||
                collider.GetComponentInParent<DamageHero>() != null;
        }

        private string Classify(Collider2D collider, HealthManager boss)
        {
            if (collider == null)
                return "none";

            string name = collider.gameObject.name.ToLowerInvariant();
            string path = GetPath(collider.transform).ToLowerInvariant();

            if (collider.GetComponentInParent<HeroController>() != null && path.Contains("/attacks/"))
                return "hero_attack";

            if (IsStaticStageHazardPath(path))
                return "stage_hazard";

            if (IsStandaloneBossAttackPath(collider, path))
                return "boss_attack";

            if (collider.gameObject.layer == 22 ||
                path.Contains("hit") ||
                path.Contains("slash") ||
                path.Contains("multihit") ||
                path.Contains("bodycatcher") ||
                path.Contains("damager") ||
                path.Contains("projectile"))
            {
                return collider.GetComponentInParent<HealthManager>() == boss
                    ? "boss_attack"
                    : "named_hitbox";
            }

            if (collider.GetComponentInParent<HeroController>() != null)
                return "hero";

            HealthManager healthManager = collider.GetComponent<HealthManager>() ?? collider.GetComponentInParent<HealthManager>();
            if (healthManager != null)
                return healthManager == boss ? "boss_or_boss_hurtbox" : "enemy_or_hurtbox";

            if (collider.GetComponent<DamageHero>() != null || collider.GetComponentInParent<DamageHero>() != null)
                return "damage_hero";

            if (name.Contains("damager") || name.Contains("slash") || name.Contains("hit"))
                return "named_hitbox";

            return collider.isTrigger ? "trigger" : "solid_or_terrain";
        }

        private bool IsStandaloneBossAttackPath(Collider2D collider, string path)
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

        private bool IsStaticStageHazardPath(string path)
        {
            return path.Contains("steam hazard") ||
                path.Contains("steam damage collider");
        }

        private string GetPath(Transform transform)
        {
            if (transform == null)
                return "";

            StringBuilder builder = new StringBuilder(transform.name);
            Transform parent = transform.parent;
            while (parent != null)
            {
                builder.Insert(0, parent.name + "/");
                parent = parent.parent;
            }

            return builder.ToString();
        }
    }
}
