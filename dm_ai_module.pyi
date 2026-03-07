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

class PlayerMode(IntEnum):
	AI = 0
	HUMAN = 1

# 再発防止: ActionType / Action は C++ レガシースタブ。削除済み。
# 新規コードは CommandType / CommandDef を使用すること。

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
	player_modes: List[PlayerMode]
	def __init__(self, *args: Any, **kwargs: Any) -> None: ...
	def setup_test_duel(self) -> None: ...
	def is_human_player(self, player_id: int) -> bool: ...
	def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int) -> None: ...
	def set_deck(self, player_id: int, deck: List[int]) -> None: ...
	def initialize_card_stats(self, *args: Any, **kwargs: Any) -> None: ...
	def clone(self) -> 'GameState': ...
	def get_card_instance(self, instance_id: int) -> Any: ...
	def get_zone(self, player_id: int, zone: Any) -> List[Any]: ...

class GameInstance:
	def __init__(self, seed: int, card_db: Any) -> None: ...
	state: GameState
	def start_game(self) -> None: ...
	def resolve_command(self, cmd: 'CommandDef') -> None: ...
	def step(self) -> bool: ...
	def undo(self) -> None: ...
	def reset_with_scenario(self, config: Any) -> None: ...
	def initialize_card_stats(self, *args: Any, **kwargs: Any) -> None: ...
	# 再発防止: resolve_action は C++ 側にバインドされていない。resolve_command を使用すること。

class GameResult:
	NONE: int
	P1_WIN: int
	P2_WIN: int
	DRAW: int

# フェーズ4: ResolutionPriority — 解決優先度（小さいほど先に解決）
# 再発防止: REPLACEMENT < INTERRUPT < NORMAL の順。
#   S・トリガー等の割り込みは INTERRUPT を指定すること。
class ResolutionPriority(IntEnum):
    REPLACEMENT = 0  # 置換効果（最優先）
    INTERRUPT   = 1  # 割り込み型: S・トリガー, G・ストライク, 忍者ストライク
    NORMAL      = 2  # 通常の誘発型能力（APNAP 順で解決）

# ── CommandDef / CommandType ─────────────────────────────────────────────────
# 再発防止: CommandType は dm/core/card_json_types.hpp の enum class。
#           Python 側は必ず CommandDef を使い、Action / dict は使わないこと。

class CommandType(IntEnum):
    # フェーズ自動遷移用プリミティブ
    TRANSITION = 0
    MUTATE = 1
    FLOW = 2
    QUERY = 3
    # マクロ
    DRAW_CARD = 4
    DISCARD = 5
    DESTROY = 6
    BOOST_MANA = 7
    TAP = 8
    UNTAP = 9
    BREAK_SHIELD = 10
    SHIELD_TRIGGER = 11
    # 移動系
    MOVE_CARD = 12
    SEND_TO_MANA = 13
    PLAYER_MANA_CHARGE = 14
    MANA_CHARGE = 15       # メインの「マナチャージ」コマンド
    # 攻撃/ブロック
    ATTACK_PLAYER = 16
    ATTACK_CREATURE = 17
    BLOCK = 18
    # カードプレイ（クリーチャー召喚 / 呪文詠唱）
    PLAY_FROM_ZONE = 19    # 手札からのプレイ（creature & spell 兼用）
    CAST_SPELL = 20
    # その他
    PASS = 21
    SELECT_TARGET = 22
    CHOICE = 23
    NONE = 24

class CommandDef:
    """C++ CommandDef の Python バインディング。
    再発防止: source_instance_id / target_instance_id は存在しない。
              正しいフィールド名: instance_id, target_instance, owner_id
    """
    type: CommandType
    instance_id: int         # ソース（主語）カードの instance_id
    target_instance: int     # ターゲットカードの instance_id
    owner_id: int            # 操作プレイヤー ID
    amount: int              # 枚数・値など
    slot_index: int          # 手札インデックス等
    target_slot_index: int
    str_param: str
    def __init__(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...

class IntentGenerator:
    """合法コマンドリストを生成する C++ ユーティリティ。
    再発防止: generate_legal_actions は後方互換エイリアス。
              新規コードは必ず generate_legal_commands を使う。
    """
    @staticmethod
    def generate_legal_commands(
        state: GameState,
        card_db: Any,
    ) -> List[CommandDef]: ...
    # deprecated alias — 新規コードでは使用しないこと
    @staticmethod
    def generate_legal_actions(
        state: GameState,
        card_db: Any,
    ) -> List[CommandDef]: ...
