using BepInEx;
using BepInEx.Configuration;
using HarmonyLib;
using UnityEngine;
using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine.SceneManagement;

namespace SilksongRL
{
    [BepInPlugin("silksongrl", "SilksongRL", "1.0.0")]
    public class RLManager : BaseUnityPlugin
    {
        // Config entries
        private ConfigEntry<string> configHost;
        private ConfigEntry<int> configPort;
        private ConfigEntry<string> configTargetBoss;
        private ConfigEntry<float> configStepInterval;
        private ConfigEntry<float> configDecisionInterval;
        private ConfigEntry<bool> configStepMode;
        private ConfigEntry<int> configStepModeFrames;
        private ConfigEntry<float> configF5RetryInterval;
        private ConfigEntry<float> configOpeningWarmupSeconds;
        private ConfigEntry<bool> configEvalMode;
        private ConfigEntry<int> configEvalStatsWindow;
        private ConfigEntry<string> configDemoRoot;
        private ConfigEntry<bool> configHitboxDebug;
        private ConfigEntry<bool> configFsmDebug;

        public static bool isAgentControlEnabled = false;
        private bool isInEval;

        // Hero and Boss references (tracked via Harmony patches)
        public static HeroController Hero { get; private set; }
        public static HealthManager Boss { get; private set; }
        
        // Static logger reference for use in Harmony patches and other classes
        public static BepInEx.Logging.ManualLogSource StaticLogger;

        private SocketClient client;
        private float stepInterval;
        private float decisionInterval;
        private bool stepModeEnabled;
        private int stepModeFrames;
        private float openingWarmupSeconds;
        private Coroutine stepModeCoroutine;

        private static IBossEncounter currentEncounter;
        public static IBossEncounter CurrentEncounter => currentEncounter;
        public static string CurrentTargetBossKey { get; private set; }

        private TrainingEpisodeManager episodeManager;
        private HitboxDebugLogger hitboxDebugLogger;
        private BossFsmDebugLogger bossFsmDebugLogger;

        private float[] previousObservations;
        private Action previousAction;
        private int[] previousActionMask;
        private int[] previousTeacherAction;
        private bool hasPreviousStep = false;
        private int whoDied = -1; // 0: Hornet, 1: Boss (same use as above ^^^)

        private bool isProcessingStep = false;
        private bool isStoringTerminalTransition = false;
        private int evalStatsWindow = 25;
        private int evalEpisodeCount = 0;
        private int evalTotalWins = 0;
        private float evalTotalBossHpPercentSum = 0f;
        private int evalWindowEpisodes = 0;
        private int evalWindowWins = 0;
        private float evalWindowBossHpPercentSum = 0f;
        private float evalBestWindowBossHpPercent = float.PositiveInfinity;

        public static Action currentAction = new Action();

        private float lastStepTime = 0f;
        private float nextDecisionTime = -1f;
        private Action heldDecisionAction = null;
        private MacroAction lastLoggedOption = (MacroAction)(-1);
        private int lastLoggedActionIndex = -1;
        private float lastOptionLogTime = -1f;
        private const float OptionLogRepeatSeconds = 1.0f;
        private const float DefaultOpeningWarmupSeconds = 0f;
        private const float OpeningWarmupLeftTolerance = 1.0f;
        private float openingWarmupUntil = -1f;
        private bool loggedOpeningWarmup = false;

        private void Awake()
        {
            StaticLogger = Logger;
            StaticLogger.LogInfo("SilksongRL Mod loaded.");

            SceneManager.sceneLoaded += OnSceneLoaded;

            configHost = Config.Bind("Connection", "Host", "localhost", 
                "Server hostname to connect to");
            configPort = Config.Bind("Connection", "Port", 8000, 
                "Server port to connect to");
            configTargetBoss = Config.Bind("Training", "TargetBoss", "Lace_1",
                "Target boss encounter (e.g., Lace_1)");
            configStepInterval = Config.Bind("Training", "StepInterval", 0.05f,
                "Time interval between observation/reward samples in seconds");
            configDecisionInterval = Config.Bind("Training", "DecisionInterval", 0.05f,
                "Minimum time between new agent decisions in seconds. The previous high-level action is repeated between decisions.");
            configStepMode = Config.Bind("Training", "StepMode", true,
                "If true, pause the game between RL decisions and advance a fixed number of physics frames after each action.");
            configStepModeFrames = Config.Bind("Training", "StepModeFrames", 3,
                "Physics frames advanced per RL sample when StepMode is enabled.");
            configF5RetryInterval = Config.Bind("Training", "F5RetryInterval", 5f,
                "Minimum seconds between repeated F5 reload attempts during automatic episode reset.");
            configOpeningWarmupSeconds = Config.Bind("Training", "OpeningWarmupSeconds", DefaultOpeningWarmupSeconds,
                "Seconds to force a short safe opening move after savestate reload. Set to 0 to disable.");
            configEvalMode = Config.Bind("Training", "EvalMode", false,
                "If true, runs in evaluation mode (no training, just inference)");
            configEvalStatsWindow = Config.Bind("Training", "EvalStatsWindow", 25,
                "Number of evaluation episodes per printed stats window when EvalMode is true");
            configDemoRoot = Config.Bind("Recorder", "DemoRoot", "",
                "Directory where human demonstration JSONL files are saved. Empty uses the Unity persistent-data directory.");
            configHitboxDebug = Config.Bind("Debug", "HitboxDebug", false,
                "Initial state for F7 collider debug logging. Press F7 in-game to toggle relevant Collider2D bounds every 0.5 seconds.");
            configFsmDebug = Config.Bind("Debug", "BossFsmDebug", false,
                "Initial state for F8 boss FSM logging. Press F8 in-game to toggle PlayMakerFSM active-state logging every 0.5 seconds.");
            
            stepInterval = configStepInterval.Value;
            decisionInterval = Mathf.Max(stepInterval, configDecisionInterval.Value);
            stepModeEnabled = configStepMode.Value;
            stepModeFrames = Mathf.Max(1, configStepModeFrames.Value);
            openingWarmupSeconds = Mathf.Max(0f, configOpeningWarmupSeconds.Value);
            isInEval = configEvalMode.Value;
            evalStatsWindow = Mathf.Max(1, configEvalStatsWindow.Value);
            gameObject.AddComponent<DemonstrationRecorder>().Initialize(configDemoRoot.Value, stepInterval);
            hitboxDebugLogger = gameObject.AddComponent<HitboxDebugLogger>();
            hitboxDebugLogger.Enabled = configHitboxDebug.Value;
            StaticLogger.LogInfo($"[HitboxDebug] Initial state: {(hitboxDebugLogger.Enabled ? "enabled" : "disabled")} (press F7 to toggle)");
            bossFsmDebugLogger = gameObject.AddComponent<BossFsmDebugLogger>();
            bossFsmDebugLogger.Enabled = configFsmDebug.Value;
            StaticLogger.LogInfo($"[BossFSM] Initial state: {(bossFsmDebugLogger.Enabled ? "enabled" : "disabled")} (press F8 to toggle)");
            
            var harmony = new Harmony("silksongrl");
            harmony.PatchAll();
            
            SocketConfig socketConfig = new SocketConfig
            {
                Host = configHost.Value,
                Port = configPort.Value,
                Timeout = 10f,
                MaxReconnectAttempts = 5,
                ReconnectDelay = 1f
            };
            client = new SocketClient(socketConfig);
            StaticLogger.LogInfo($"[RL] Using socket transport: {configHost.Value}:{configPort.Value}");
        }

        private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            RLManager.StaticLogger?.LogInfo($"[ScreenCapture] Scene loaded: {scene.name}");

            if (scene.name == "Pre_Menu_Loader" || scene.name == "Pre_Menu_Intro") return;

            // Initialize encounter based on config
            CurrentTargetBossKey = configTargetBoss.Value;
            currentEncounter = CreateEncounter(CurrentTargetBossKey);
            if (currentEncounter == null)
            {
                StaticLogger.LogError($"[RL] Unknown boss encounter: {configTargetBoss.Value}");
                return;
            }

            // Initialize screen capture updater for hybrid encounters
            // This isn't done in Awake because main camera isn't available yet
            if (currentEncounter.GetObservationType() == ObservationType.Hybrid)
            {
                var screenCapture = currentEncounter.GetScreenCapture();
                if (screenCapture != null)
                {
                    var updater = gameObject.AddComponent<ScreenCaptureUpdater>();
                    updater.Initialize(screenCapture);
                    StaticLogger.LogInfo("[RL] Screen capture updater initialized for hybrid observation");
                }
            }
            
            episodeManager = new TrainingEpisodeManager(currentEncounter, configF5RetryInterval.Value);
            episodeManager.OnSimulateKeyPress = SimulateKeyPress;
            episodeManager.OnResetComplete = ResetRL;

            StaticLogger.LogInfo($"[RL] Initialized with encounter: {currentEncounter.GetEncounterName()}");
            StaticLogger.LogInfo($"[RL] Observation size: {currentEncounter.GetObservationSize()}");
            StaticLogger.LogInfo(ActionManager.UsesRawActionSpace()
                ? $"[RL] Action space: Raw key branches ({string.Join(",", ActionManager.GetActionSpaceShape())})"
                : $"[RL] Action space: Macro options ({ActionManager.MacroActionCount})");
            StaticLogger.LogInfo($"[RL] Action space shape: [{string.Join(",", ActionManager.GetActionSpaceShape())}], joint actions: {ActionManager.GetActionCount()}");
            StaticLogger.LogInfo($"[RL] Step interval: {stepInterval:0.###}s, decision interval: {decisionInterval:0.###}s, step mode: {stepModeEnabled}, frames: {stepModeFrames}");
            StaticLogger.LogInfo($"[RL] Reset tuning: F5 retry {configF5RetryInterval.Value:0.###}s, opening warmup {openingWarmupSeconds:0.###}s");
            StaticLogger.LogInfo($"[RL] Mode: {(isInEval ? "Evaluation" : "Training")}");
            if (isInEval)
            {
                StaticLogger.LogInfo($"[EvalStats] Window size: {evalStatsWindow} episodes");
            }
            
            _ = InitializeClientAsync();

            SceneManager.sceneLoaded -= OnSceneLoaded;
            //这个设计有一个潜在问题：如果Awake（）在一次游戏启动只调用一次的话，
            //如果你切换到另一个场景、换 Boss、重载游戏，它未必会重新初始化 encounter。这个项目大概率假设你一次打开游戏只训练一个固定 Boss，这是个待优化的点。
        }

        private IBossEncounter CreateEncounter(string bossName)
        {
            switch ((bossName ?? "").Trim())
            {
                case "Lace_1":
                case "lace_1":
                case "Lace Boss1":
                case "LaceBoss1":
                    return new LaceEncounter();
                case "Lace_2":
                case "lace_2":
                case "Lace Boss2 New":
                case "LaceBoss2New":
                    return new LaceSecondEncounter();
                case "Savage_Beastfly":
                case "savage_beastfly":
                    return new SavageBeastflyEncounter();
                default:
                    return null;
            }
        }

        private void OnDestroy()
        {
            StopStepModeLoop();
            client?.Disconnect();
            StaticLogger.LogInfo("[RL] Client disconnected");
        }

        private async Task InitializeClientAsync()
        {
            try
            {
                // Connect first (no-op for HTTP, establishes connection for sockets)
                bool connected = await client.ConnectAsync();
                if (!connected)
                {
                    StaticLogger.LogError("[RL] Failed to connect to server!");
                    return;
                }

                string bossName = currentEncounter.GetEncounterName();
                int obsSize = currentEncounter.GetObservationSize();
                int[] actionSpaceShape = ActionManager.GetActionSpaceShape();
                ObservationType obsType = currentEncounter.GetObservationType();
                int vectorObsSize = currentEncounter.GetVectorObservationSize();
                var (visualWidth, visualHeight) = currentEncounter.GetVisualObservationSize();
                
                StaticLogger.LogInfo($"[RL] Initializing client for boss: {bossName}");
                StaticLogger.LogInfo($"[RL]   Observation size: {obsSize}, type: {obsType}, vector size: {vectorObsSize}");
                if (obsType == ObservationType.Hybrid)
                StaticLogger.LogInfo($"[RL]   Visual size: {visualWidth}x{visualHeight}");
                
                var response = await client.InitializeAsync(bossName, obsSize, actionSpaceShape, obsType, vectorObsSize, visualWidth, visualHeight, isInEval);
                
                if (response != null && response.initialized)
                {
                    StaticLogger.LogInfo($"[RL] Client initialized successfully. Checkpoint loaded: {response.checkpoint_loaded}");
                }
                else
                {
                    StaticLogger.LogError("[RL] Client initialization failed!");
                }
            }
            catch (Exception e)
            {
                StaticLogger.LogError($"[RL] Error initializing client: {e.Message}");
            }
        }

        private void Update()
        {
            // Toggle control when pressing P
            if (Input.GetKeyDown(KeyCode.P))
            {
                isAgentControlEnabled = !isAgentControlEnabled;
                if (isAgentControlEnabled)
                {
                    ResetRL();
                    currentEncounter?.ResetObservationHistory();
                    StartStepModeLoopIfNeeded();
                }
                else
                {
                    StopStepModeLoop();
                }
                
                StaticLogger.LogInfo($"[RL] Agent control {(isAgentControlEnabled ? "enabled" : "disabled")}. Hero: {(Hero != null ? "Found" : "Not found")}, Boss: {(Boss != null ? "Found" : "Not found")}");
            }
            
            // Log resolution diagnostics when pressing L
            if (Input.GetKeyDown(KeyCode.L))
            {
                gameObject.AddComponent<ScreenCaptureTest>();
                ResolutionDiagnostics.LogResolutionInfo(StaticLogger);
                ResolutionDiagnostics.CheckForPotentialIssues(StaticLogger);
            }

            if (Input.GetKeyDown(KeyCode.F7) && hitboxDebugLogger != null)
            {
                hitboxDebugLogger.Enabled = !hitboxDebugLogger.Enabled;
                configHitboxDebug.Value = hitboxDebugLogger.Enabled;
                StaticLogger.LogInfo($"[HitboxDebug] {(hitboxDebugLogger.Enabled ? "enabled" : "disabled")} by F7");
            }

            if (Input.GetKeyDown(KeyCode.F8) && bossFsmDebugLogger != null)
            {
                bossFsmDebugLogger.Enabled = !bossFsmDebugLogger.Enabled;
                configFsmDebug.Value = bossFsmDebugLogger.Enabled;
                bossFsmDebugLogger.ResetCache();
                StaticLogger.LogInfo($"[BossFSM] {(bossFsmDebugLogger.Enabled ? "enabled" : "disabled")} by F8");
            }

            hitboxDebugLogger?.Tick(Hero, Boss);
            bossFsmDebugLogger?.Tick(Boss);
        }

        private GUIStyle pingStyle;
        
        private void OnGUI()
        {
            if (client == null) return;
            
            if (pingStyle == null)
            {
                pingStyle = new GUIStyle(GUI.skin.label);
                pingStyle.normal.textColor = Color.white;
            }
            
            float ping = client.lastPingMs;
            string pingText = $"{ping:F0} ms";
            
            float padding = 10f;
            float width = 40f;
            float height = 25f;
            Rect rect = new Rect(Screen.width - width - padding, padding, width, height);
            
            GUI.color = Color.black;
            GUI.Label(new Rect(rect.x + 1, rect.y + 1, rect.width, rect.height), pingText, pingStyle);
            
            GUI.color = Color.white;
            GUI.Label(rect, pingText, pingStyle);
        }

        private void FixedUpdate()
        {
            if (!isAgentControlEnabled)
            {
                ActionManager.CancelSustainedOption();
                currentAction = new Action();
                return;
            }

            if (stepModeEnabled)
                return;

            if (!ProcessEpisodeAndResetState())
                return;

            if (ProcessOpeningWarmup())
                return;

            // Step on a **frame independent** fixed time interval
            if (Time.fixedTime - lastStepTime >= stepInterval)
            {
                lastStepTime = Time.fixedTime;
                _ = StepRLAsync();
            }
        }

        private bool ProcessEpisodeAndResetState()
        {
            if (currentEncounter == null || episodeManager == null)
            {
                currentAction = new Action();
                return false;
            }

            if (currentAction == null)
            {
                currentAction = new Action();
            }

            var previousState = episodeManager.CurrentState;
            episodeManager.UpdateEpisodeState(Hero, Boss);
            
            if (previousState == TrainingEpisodeManager.EpisodeState.Training && 
                (episodeManager.CurrentState == TrainingEpisodeManager.EpisodeState.HeroDead || 
                 episodeManager.CurrentState == TrainingEpisodeManager.EpisodeState.BossDead ||
                 episodeManager.CurrentState == TrainingEpisodeManager.EpisodeState.HeroStuck))
            {
                // HeroDead or HeroStuck = hero died (0), BossDead = boss died (1)
                whoDied = (episodeManager.CurrentState == TrainingEpisodeManager.EpisodeState.BossDead) ? 1 : 0;
                if (isInEval)
                {
                    RecordEvalEpisodeEnd(episodeManager.CurrentState);
                    StaticLogger.LogInfo("[RL] Eval episode ended - no transition stored");
                }
                else
                {
                    StaticLogger.LogInfo("[RL] Episode ended - will store final transition with done=true");
                    _ = StoreTerminalTransitionAsync(whoDied);
                }
            }

            if (episodeManager.HandleResetSequence(Hero, Boss))
            {
                currentAction = new Action();
                ActionManager.CancelSustainedOption();
                return false;
            }

            return true;
        }

        private void RecordEvalEpisodeEnd(TrainingEpisodeManager.EpisodeState terminalState)
        {
            bool won = terminalState == TrainingEpisodeManager.EpisodeState.BossDead;
            float bossHpPercent = won ? 0f : GetCurrentBossHpPercent();

            evalEpisodeCount++;
            evalWindowEpisodes++;
            if (won)
            {
                evalTotalWins++;
                evalWindowWins++;
            }

            evalTotalBossHpPercentSum += bossHpPercent;
            evalWindowBossHpPercentSum += bossHpPercent;

            string reason = terminalState == TrainingEpisodeManager.EpisodeState.BossDead
                ? "boss_defeated"
                : terminalState == TrainingEpisodeManager.EpisodeState.HeroStuck
                    ? "hero_stuck"
                    : "hero_dead";

            StaticLogger.LogInfo(
                $"[EvalStats] Episode {evalEpisodeCount}: {(won ? "WIN" : "LOSS")}, " +
                $"reason {reason}, boss remaining HP {bossHpPercent:0.0}%");

            if (evalWindowEpisodes >= evalStatsWindow)
            {
                float avgBossHp = evalWindowBossHpPercentSum / evalWindowEpisodes;
                float winRate = evalWindowWins / (float)evalWindowEpisodes;
                float totalAvgBossHp = evalTotalBossHpPercentSum / evalEpisodeCount;
                float totalWinRate = evalTotalWins / (float)evalEpisodeCount;
                bool newBestWindow = avgBossHp < evalBestWindowBossHpPercent;
                if (newBestWindow)
                {
                    evalBestWindowBossHpPercent = avgBossHp;
                }

                StaticLogger.LogInfo(
                    $"[EvalStats] Recent {evalWindowEpisodes} episode stats after {evalEpisodeCount} episodes: " +
                    $"{evalWindowWins}/{evalWindowEpisodes} wins ({winRate * 100f:0.0}%), " +
                    $"avg boss remaining HP {avgBossHp:0.0}%" +
                    $"{(newBestWindow ? " (new eval best)" : "")}; " +
                    $"overall {evalTotalWins}/{evalEpisodeCount} wins ({totalWinRate * 100f:0.0}%), " +
                    $"overall avg boss remaining HP {totalAvgBossHp:0.0}%");

                evalWindowEpisodes = 0;
                evalWindowWins = 0;
                evalWindowBossHpPercentSum = 0f;
            }
        }

        private float GetCurrentBossHpPercent()
        {
            if (Boss == null || currentEncounter == null)
                return 100f;

            float maxHp = Mathf.Max(1f, currentEncounter.GetMaxHP());
            return Mathf.Clamp01(Boss.hp / maxHp) * 100f;
        }

        private bool ProcessOpeningWarmup()
        {
            if (IsOpeningWarmupActive())
            {
                ClearPreviousTransitionState();
                ActionManager.CancelSustainedOption();
                ClearHeldDecisionAction();
                currentAction = GetOpeningWarmupAction();
                lastStepTime = Time.fixedTime;

                if (!loggedOpeningWarmup)
                {
                    StaticLogger.LogInfo("[RL] Opening warmup active: left movement disabled, transitions skipped");
                    loggedOpeningWarmup = true;
                }

                return true;
            }

            return false;
        }

        private void StartStepModeLoopIfNeeded()
        {
            if (!stepModeEnabled || stepModeCoroutine != null)
                return;

            stepModeCoroutine = StartCoroutine(StepModeLoop());
            StaticLogger.LogInfo($"[RL] Step mode started: {stepModeFrames} physics frames per sample");
        }

        private void StopStepModeLoop()
        {
            if (stepModeCoroutine != null)
            {
                StopCoroutine(stepModeCoroutine);
                stepModeCoroutine = null;
            }

            Time.timeScale = 1f;
        }

        private IEnumerator StepModeLoop()
        {
            while (isAgentControlEnabled)
            {
                if (!ProcessEpisodeAndResetState())
                {
                    Time.timeScale = 1f;
                    yield return new WaitForFixedUpdate();
                    continue;
                }

                if (ProcessOpeningWarmup())
                {
                    Time.timeScale = 1f;
                    yield return new WaitForFixedUpdate();
                    continue;
                }

                Time.timeScale = 0f;
                Task stepTask = StepRLAsync();
                while (!stepTask.IsCompleted)
                {
                    yield return null;
                }

                if (stepTask.IsFaulted)
                {
                    StaticLogger.LogError($"[RL] Step mode task failed: {stepTask.Exception?.GetBaseException().Message}");
                }

                Time.timeScale = 1f;
                for (int i = 0; i < stepModeFrames; i++)
                {
                    yield return new WaitForFixedUpdate();
                }
            }

            Time.timeScale = 1f;
            stepModeCoroutine = null;
        }

        private async Task StepRLAsync()
        {
            if (isProcessingStep) return;

            isProcessingStep = true;

            try
            {
                if (Hero == null)
                {
                    StaticLogger.LogWarning("[RL] Hero is null - waiting for hero to spawn");
                    return;
                }
                if (Boss == null)
                {
                    StaticLogger.LogWarning("[RL] Boss is null - waiting for boss to spawn");
                    return;
                }
                
                float[] currentObservations = currentEncounter.ExtractObservationArray(Hero, Boss);
                if (currentObservations == null)
                {
                    StaticLogger.LogWarning("[RL] Observation extraction returned null");
                    return;
                }
                int[] currentActionMask = currentEncounter.GetActionMask(Hero, Boss);
                int[] currentTeacherAction = ActionManager.TeacherMacroToActionArray(
                    currentEncounter.GetTeacherAction(Hero, Boss, currentActionMask),
                    Hero,
                    Boss,
                    currentEncounter);

                // Store transition from previous step (training mode only)
                if (!isInEval && hasPreviousStep && previousObservations != null)
                {
                    float reward = currentEncounter.CalculateReward(previousObservations, currentObservations, previousAction, -1);
                    RewardComponents rewardComponents = currentEncounter.GetLastRewardComponents();

                    // Run socket call off the main thread
                    await Task.Run(async () =>
                    {
                        await client.StoreTransitionAsync(previousObservations, previousAction, previousActionMask, currentActionMask, previousTeacherAction, reward, rewardComponents, currentObservations, false).ConfigureAwait(false);
                    });
                }

                Action action;
                bool usingSustainedAction = ActionManager.TryGetSustainedAction(Hero, Boss, currentEncounter, out action);
                if (usingSustainedAction && action != null && currentActionMask != null)
                {
                    int actionIndex = ActionManager.ActionToIndex(action);
                    if (actionIndex >= 0 && actionIndex < currentActionMask.Length)
                    {
                        currentActionMask[actionIndex] = 1;
                    }
                }

                if (!usingSustainedAction)
                {
                    if (CanReuseHeldDecisionAction(currentActionMask))
                    {
                        action = ActionManager.ActionIndexToAction(heldDecisionAction.actionIndex, Hero, Boss, currentEncounter);
                    }
                    else
                    {
                        // Get action from the RL agent
                        action = await client.GetActionAsync(currentObservations, currentActionMask);

                        HoldDecisionAction(action);
                    }

                    if (ActionManager.ShouldBeginSustainedOption(action))
                    {
                        ActionManager.BeginSustainedOption(action);
                    }
                }

                if (action != null)
                {
                    ActionManager.PrepareActionForExecution(action);

                    int executedActionIndex = ActionManager.ActionToIndex(action);
                    if (currentActionMask != null &&
                        executedActionIndex >= 0 &&
                        executedActionIndex < currentActionMask.Length)
                    {
                        currentActionMask[executedActionIndex] = 1;
                    }

                    currentAction = action;
                    LogCurrentOption(action, usingSustainedAction);

                    // Only track previous state during training (needed for storing transitions)
                    if (!isInEval)
                    {
                        previousObservations = currentObservations;
                        previousAction = action;
                        previousActionMask = currentActionMask;
                        previousTeacherAction = currentTeacherAction;
                        hasPreviousStep = true;
                    }
                }
            }
            catch (Exception e)
            {
                StaticLogger.LogError($"[RL] Error in StepRL: {e.Message}");
            }
            finally
            {
                isProcessingStep = false;
            }
        }

        private bool CanReuseHeldDecisionAction(int[] currentActionMask)
        {
            if (heldDecisionAction == null)
                return false;

            if (Time.fixedTime >= nextDecisionTime)
                return false;

            int actionIndex = heldDecisionAction.actionIndex;
            return currentActionMask != null &&
                actionIndex >= 0 &&
                actionIndex < currentActionMask.Length &&
                currentActionMask[actionIndex] != 0;
        }

        private void HoldDecisionAction(Action action)
        {
            if (action == null)
            {
                ClearHeldDecisionAction();
                return;
            }

            heldDecisionAction = action;
            nextDecisionTime = Time.fixedTime + decisionInterval;
        }

        private void ClearHeldDecisionAction()
        {
            heldDecisionAction = null;
            nextDecisionTime = -1f;
        }

        private void LogCurrentOption(Action action, bool sustained)
        {
            if (action == null)
                return;

            bool optionChanged = action.macro != lastLoggedOption || action.actionIndex != lastLoggedActionIndex;
            bool shouldRepeat = Time.unscaledTime - lastOptionLogTime >= OptionLogRepeatSeconds;
            if (!optionChanged && !shouldRepeat)
                return;

            lastLoggedOption = action.macro;
            lastLoggedActionIndex = action.actionIndex;
            lastOptionLogTime = Time.unscaledTime;
            StaticLogger.LogInfo($"[RL] Current action: {ActionManager.GetActionName(action)} / {action.macro}{(sustained ? " (sustained)" : "")}");
        }

        private async Task StoreTerminalTransitionAsync(int terminalWhoDied)
        {
            if (isInEval || isStoringTerminalTransition || !hasPreviousStep || previousObservations == null || previousAction == null)
                return;

            isStoringTerminalTransition = true;

            try
            {
                float[] terminalObservations = currentEncounter.CreateTerminalObservation(previousObservations, terminalWhoDied);
                if (terminalObservations == null)
                    return;

                float reward = currentEncounter.CalculateReward(previousObservations, terminalObservations, previousAction, terminalWhoDied);
                RewardComponents rewardComponents = currentEncounter.GetLastRewardComponents();

                await Task.Run(async () =>
                {
                    await client.StoreTransitionAsync(previousObservations, previousAction, previousActionMask, previousActionMask, previousTeacherAction, reward, rewardComponents, terminalObservations, true).ConfigureAwait(false);
                }).ConfigureAwait(false);

                StaticLogger.LogInfo("[RL] Stored final transition with done=true");
                previousObservations = null;
                previousAction = null;
                previousActionMask = null;
                previousTeacherAction = null;
                hasPreviousStep = false;
                whoDied = -1;
                ClearHeldDecisionAction();
            }
            catch (Exception e)
            {
                StaticLogger.LogError($"[RL] Error storing terminal transition: {e.Message}");
            }
            finally
            {
                isStoringTerminalTransition = false;
            }
        }


        private void ResetRL()
        {
            // Clear current action and processing flag
            currentAction = GetOpeningWarmupAction();
            ActionManager.CancelSustainedOption();
            isProcessingStep = false;
            isStoringTerminalTransition = false;
            ClearPreviousTransitionState();
            ClearHeldDecisionAction();
            whoDied = -1;
            openingWarmupUntil = ShouldUseOpeningWarmup()
                ? Time.fixedTime + openingWarmupSeconds
                : -1f;
            loggedOpeningWarmup = false;
            currentEncounter?.ResetObservationHistory();
            lastLoggedActionIndex = -1;
        }

        private bool IsOpeningWarmupActive()
        {
            return isAgentControlEnabled &&
                openingWarmupSeconds > 0f &&
                Time.fixedTime <= openingWarmupUntil &&
                ShouldUseOpeningWarmup();
        }

        private bool ShouldUseOpeningWarmup()
        {
            if (Hero == null || currentEncounter == null)
                return false;

            return Hero.transform.position.x < currentEncounter.GetPreferredCenterX() - OpeningWarmupLeftTolerance;
        }

        private Action GetOpeningWarmupAction()
        {
            MoveDirection move = MoveDirection.None;
            if (Hero != null && currentEncounter != null && Hero.transform.position.x < currentEncounter.GetPreferredCenterX())
            {
                move = MoveDirection.Right;
            }

            Action action = new Action
            {
                macro = MacroAction.MoveToCenterSafely,
                move = move,
                look = LookDirection.None,
                jump = false,
                attack = false,
                dash = false,
                needle = false,
                bind = false,
                tool = ToolAction.None,
                optionStarted = false
            };
            action.actionIndex = ActionManager.ActionToIndex(action);
            return action;
        }

        private void ClearPreviousTransitionState()
        {
            previousObservations = null;
            previousAction = null;
            previousActionMask = null;
            previousTeacherAction = null;
            hasPreviousStep = false;
        }

        // Static flag for F5 simulation
        private const float F5SimulationDuration = 0.2f;
        private static float simulateF5PressUntil = -1f;

        public static bool IsSimulatingF5Press()
        {
            return Time.unscaledTime <= simulateF5PressUntil;
        }

        private void SimulateKeyPress(KeyCode key)
        {
            if (key == KeyCode.F5)
            {
                simulateF5PressUntil = Mathf.Max(simulateF5PressUntil, Time.unscaledTime + F5SimulationDuration);
                StaticLogger.LogInfo("[RL] Simulating F5 key press");
            }
        }

        /// <summary>
        /// Harmony patch to automatically catch Hero spawns.
        /// </summary>
        [HarmonyPatch(typeof(HeroController), "Awake")]
        public static class HeroController_Awake_Patch
        {
            static void Postfix(HeroController __instance)
            {
                Hero = __instance;
                StaticLogger.LogInfo("[RL] Hero found and assigned (Harmony patch)");
            }
        }

        /// <summary>
        /// Harmony patch to automatically catch Boss spawns.
        /// </summary>
        [HarmonyPatch(typeof(HealthManager), "Awake")]
        public static class HealthManager_Awake_Patch
        {
            static void Postfix(HealthManager __instance)
            {
                // Only assign if we have an encounter configured and this matches
                //假如并不是对每个特定boss单独训练，则是否还要match bossname？，或者仅仅用这个match来判断是不是游戏内的boss，而不是杂兵
                //待定
                if (currentEncounter != null && currentEncounter.IsEncounterMatch(__instance))
                {
                    Boss = __instance;
                    StaticLogger.LogInfo($"[RL] Boss locked: {__instance.name} (Harmony patch)");
                }
            }
        }
        //假如需要更多信息去计算状态，奖励，动作等，可能需要patch更多信息
    }
}
