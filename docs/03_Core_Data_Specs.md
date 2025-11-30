# 3. コア・データ仕様 (Core Data Specs)

## 3.1 定数と制限 (constants.hpp)
- **MAX_HAND_SIZE**: 20
- **MAX_BATTLE_SIZE**: 20
- **MAX_MANA_SIZE**: 20
- **MAX_GRAVE_SEARCH**: 20
- **TURN_LIMIT**: 100
- **POWER_INFINITY**: 32000

## 3.2 カードデータ構造 (card_def.hpp)
- **ID管理**: CardID (uint16_t).
- **Keywords**:
    - **Basic**: `SPEED_ATTACKER`, `BLOCKER`, `SLAYER`, `EVOLUTION`.
    - **Breakers**: `DOUBLE_BREAKER`, `TRIPLE_BREAKER`.
    - **Parameterized**: `POWER_ATTACKER` (Bonus Value).
- **Filter Parsing**: CSVロード時に文字列条件をID化して保持.
- **Mode Selection**: ModalEffectGroup 構造体による複数選択管理.

## 3.3 盤面状態 (game_state.hpp)
- **Determinism**: `std::mt19937` をState内に保持し、シード値による完全再現を保証.
- **Incomplete Info**: 学習用Viewでは相手の手札・山札の中身をマスクまたはランダム化.
- **Error Handling**: 異常状態で `std::runtime_error` を送出し、即時停止.
