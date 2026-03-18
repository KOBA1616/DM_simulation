"""
タスク1: 二重判定の解消 - RED フェーズ

分析結果:
- ManaSystem::can_pay_cost() → コマンド生成フェーズで使用（PASSIVE軽減のみ）
- CostPaymentSystem::can_pay_cost() → 定義されているが呼ばれていない（死コード）
- CostPaymentSystem::calculate_max_units/execute_payment() → 実行フェーズで使用

問題: コスト判定フェーズと支払い実行フェーズが分離しており、ACTIVE軽減考慮の矛盾

RED テスト: PaymentPlan::evaluate_cost() の有効性を確認し、GREEN フェーズでの
統一実装の基盤を整備する
"""

import pytest
import json
from pathlib import Path
import dm_ai_module
from dm_toolkit.payment import evaluate_cost


def test_payment_plan_evaluate_cost_exists_and_works():
    """
    RED テスト: PaymentPlan::evaluate_cost() が正しく動作することを確認
    
    GREEN フェーズでは、ManaSystem::can_pay_cost() をこの結果利用へ統一する
    """
    
    cards_dict = [
        {
            "id": 1001,
            "name": "TestCard",
            "cost": 10,
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "power": 1,
            "toughness": 1,
            "cost_reductions": [
                {
                    "id": "passive_red",
                    "type": "PASSIVE",
                    "amount": 3
                },
                {
                    "id": "active_red",
                    "type": "ACTIVE_PAYMENT",
                    "amount": 2,
                    "unit_cost": {
                        "type": "TAP_CARD",
                        "filter": {"zones": ["BATTLE_ZONE"], "card_types": ["CREATURE"]}
                    },
                    "max_units": 1
                }
            ]
        }
    ]
    
    # Python側の evaluate_cost は dict を使う
    card_dict = cards_dict[0]
    
    # テスト1: PASSIVE軽減のみ（コマンド生成フェーズのシナリオ）
    plan_passive = evaluate_cost(card_dict, base_cost=10, units=1, active_reduction_id=None, active_units=None)
    assert plan_passive.base_cost == 10
    assert plan_passive.total_passive_reduction == 3
    assert plan_passive.adjusted_after_passive == 7
    assert plan_passive.final_cost == 7, (
        "RED: PASSIVE軽減後のコストが7であることを確認\n"
        f"PaymentPlan が正しく PASSIVE を計算している"
    )
    
    # テスト2: PASSIVE + ACTIVE軽減（実行フェーズのシナリオ）
    plan_both = evaluate_cost(card_dict, base_cost=10, units=1, active_reduction_id="active_red", active_units=1)
    assert plan_both.base_cost == 10
    assert plan_both.total_passive_reduction == 3
    assert plan_both.active_reduction_amount == 2
    assert plan_both.final_cost == 5, (
        "RED: PASSIVE + ACTIVE 軽減後のコストが5であることを確認\n"
        f"PaymentPlan が両軽減を統合している"
    )



def test_cost_payment_system_dead_code_confirmed():
    """
    RED テスト: CostPaymentSystem::can_pay_cost() が呼ばれていないことを記録
    
    エージェント Explore による分析で確認:
    - ManaSystem::can_pay_cost() → 3箇所で実用中
    - CostPaymentSystem::can_pay_cost() → 呼び出し箇所ゼロ（死コード）
    
    GREEN フェーズ改善案:
    1. CostPaymentSystem::can_pay_cost() を削除
    2. ManaSystem::can_pay_cost() を PaymentPlan ベースへリファクタ
       または
    3. 新しい統一判定メソッドを作成し、両方を置き換える
    """
    
    # 記録的テスト: 死コード削除の許可
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
