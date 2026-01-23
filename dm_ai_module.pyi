from typing import Any, Callable, List, Tuple, Dict
from enum import IntEnum

# Improved permissive stub for the C/C++ extension module. Commonly used
# APIs are given lightweight signatures to improve mypy usefulness. Keep
# most types as Any for now; refine gradually.

PhaseManager: Any
ActionGenerator: Any
EffectResolver: Any
JsonLoader: Any

class Phase(IntEnum):
    START = 0
    DRAW = 1
    MANA = 2
    MAIN = 3
    ATTACK = 4
    END = 5

class ActionType(IntEnum):
    PLAY_CARD = 1
    ATTACK_PLAYER = 2
    ATTACK_CREATURE = 3
    BLOCK_CREATURE = 4
    PASS = 5
    USE_SHIELD_TRIGGER = 6
    MANA_CHARGE = 7
    RESOLVE_EFFECT = 8
    SELECT_TARGET = 9
    TAP = 10
    UNTAP = 11
    BREAK_SHIELD = 14

class Action:
    type: ActionType
    target_player: int
    source_instance_id: int
    card_id: int
    slot_index: int
    value1: int
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class CardDatabase:
    @staticmethod
    def load(path: str = ...) -> None: ...
    @staticmethod
    def get_card(card_id: int) -> Dict[str, Any]: ...
    def __init__(self) -> None: ...

class ParallelRunner:
	def __init__(self, card_db: Any, sims: int, batch_size: int) -> None: ...
	def play_games(self, initial_states: List[Any], evaluator: Any, temp: float, add_noise: bool, threads: int, alpha: float = 0.0, collect_data: bool = False) -> List[Any]: ...
	def play_deck_matchup(self, deck_a: List[int], deck_b: List[int], num_games: int, threads: int) -> List[int]: ...

class TensorConverter:
	INPUT_SIZE: int
	@staticmethod
	def convert_to_tensor(*args: Any, **kwargs: Any) -> Any: ...
	@staticmethod
	def convert_batch_flat(*args: Any, **kwargs: Any) -> Any: ...

class ActionEncoder:
	TOTAL_ACTION_SIZE: int
	@staticmethod
	def action_to_index(action: Any) -> int: ...
	@staticmethod
	def index_to_action(idx: int) -> Any: ...

class TokenConverter:
	@staticmethod
	def get_vocab_size() -> int: ...

class DeckEvolutionConfig:
	target_deck_size: int
	mutation_rate: float

class DeckEvolution:
	def __init__(self, card_db: Any) -> None: ...
	def evolve_deck(self, parent_deck: List[int], candidate_pool: List[int], config: DeckEvolutionConfig) -> List[int]: ...
	def calculate_interaction_score(self, deck: List[int]) -> float: ...
	def get_candidates_by_civ(self, candidate_pool: List[int], civ: Any) -> List[int]: ...

class NeuralEvaluator:
	def __init__(self, card_db: Any) -> None: ...
	def set_model_type(self, model_type: Any) -> None: ...

class ModelType:
	TRANSFORMER: Any
	RESNET: Any

class CardType:
	CREATURE: Any
	SPELL: Any

def set_flat_batch_callback(cb: Callable[..., Any]) -> None: ...
def clear_flat_batch_callback() -> None: ...
def set_sequence_batch_callback(cb: Callable[..., Any]) -> None: ...
def clear_sequence_batch_callback() -> None: ...

def get_card_stats(state: Any) -> Any: ...

def __getattr__(name: str) -> Any: ...

class Player:
	hand: List[Any]
	mana_zone: List[Any]
	battle_zone: List[Any]
	graveyard: List[Any]

class GameState:
	game_over: bool
	turn_number: int
	active_player_id: int
	players: List[Player]
	current_phase: Phase
	winner: Any
	def __init__(self, *args: Any, **kwargs: Any) -> None: ...
	def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int) -> None: ...
	def set_deck(self, player_id: int, deck: List[int]) -> None: ...
	def initialize_card_stats(self, *args: Any, **kwargs: Any) -> None: ...
	def clone(self) -> 'GameState': ...
	def get_card_instance(self, instance_id: int) -> Any: ...
	def get_zone(self, player_id: int, zone: Any) -> List[Any]: ...

class GameInstance:
	def __init__(self, *args: Any, **kwargs: Any) -> None: ...
	state: GameState
	def reset_with_scenario(self, config: Any) -> None: ...
	def initialize_card_stats(self, *args: Any, **kwargs: Any) -> None: ...
	def resolve_action(self, action: Any) -> None: ...

class GameResult:
	NONE: int
	P1_WIN: int
	P2_WIN: int
	DRAW: int
