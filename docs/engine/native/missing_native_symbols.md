# Missing native symbols report

Detected references to `dm_ai_module` symbols and whether they are present.

## Present

- Action
- ActionType
- CardDatabase
- CardStub
- CardType
- Civilization
- CommandDef
- CommandSystem
- CommandType
- DataCollector
- FlowCommand
- FlowType
- GameCommand
- GameInstance
- GameResult
- GameState
- JsonLoader
- ParallelRunner
- Phase
- PhaseManager
- __file__

## Missing
- ActionDef
- ActionEncoder
- ActionGenerator
- ActionType (ambiguous / noisy matches)

- CardDefinition
- CardKeywords
- CardRegistry
- EffectActionType
- EffectDef
- EffectResolver
- EffectType
- ConditionDef
- FilterDef

- MutateCommand
- MutationType
- TransitionCommand
- TriggerType
- TokenConverter
- TensorConverter

- GameLogicSystem
- DeckEvolution
- DeckEvolutionConfig
- DeckInference

- HeuristicAgent
- HeuristicEvaluator
- NeuralEvaluator
- ModelType
- POMDPInference

- ScenarioConfig
- ScenarioExecutor

- PlayerStub
- PassiveEffect
- PassiveType

Notes:
- The original scanner found many noisy / context strings (test names, code fragments). The list above filters to likely symbol names referenced by code that are not currently exported by `dm_ai_module`.
- Many of these are higher-level native components (in C++) that should be implemented in the native binding for production. Short-term Python shims/stubs can be added to allow tests/scripts to run.
