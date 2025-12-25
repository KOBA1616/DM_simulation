from typing import Any

# Permissive stub for the C/C++ extension module used by the project.
# Many symbols are exported from the native extension; to keep mypy runs
# actionable we expose the most-used names as `Any` to avoid attr-defined
# noise. Replace specific names with precise signatures as the stubs
# are improved.

GameResult: Any
CardInstance: Any
Player: Any
GameState: Any
GameInstance: Any
PhaseManager: Any
ActionType: Any
Action: Any
ActionGenerator: Any
EffectResolver: Any
JsonLoader: Any
ParallelRunner: Any
TensorConverter: Any
ActionEncoder: Any
DevTools: Any
Zone: Any

# Additional commonly referenced symbols
EffectSystem: Any
ActionDef: Any
EffectActionType: Any
InstructionOp: Any
PipelineExecutor: Any
CardDefinition: Any
CardKeywords: Any
FilterDef: Any
DeckEvolutionConfig: Any
DeckEvolution: Any
HeuristicEvaluator: Any
CardType: Any
TokenConverter: Any
NeuralEvaluator: Any
ModelType: Any
DataCollector: Any

def set_flat_batch_callback(cb: Any) -> None: ...
def clear_flat_batch_callback() -> None: ...
def get_card_stats(state: Any) -> Any: ...

def __getattr__(name: str) -> Any: ...
