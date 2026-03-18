from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class PaymentPlan:
    base_cost: int
    adjusted_after_passive: int
    total_passive_reduction: int
    passive_ids: List[str] = field(default_factory=list)
    active_reduction_id: Optional[str] = None
    active_units: Optional[int] = None
    active_reduction_amount: int = 0
    final_cost: int = 0




def _compute_cr_reduction(cr: Dict[str, Any], units: int) -> int:
    """Compute numeric reduction contributed by a single PASSIVE cost_reduction.

    Heuristic rules (prototype):
    - If `reduction_per_unit` present: reduction = reduction_per_unit * applied_units
    - Elif `amount` present: reduction = amount
    - Elif `unit_cost` present: reduction = unit_cost * applied_units
    - applied_units = min(units, max_units) if max_units present and >0 else units
    - `min_mana_cost` acts as a floor on the final adjusted cost, not on reduction.
    """
    if not isinstance(cr, dict):
        return 0
    try:
        max_units = int(cr.get('max_units')) if cr.get('max_units') is not None else 0
    except Exception:
        max_units = 0
    applied = units
    if max_units and max_units > 0:
        applied = min(units, max_units)

    if 'reduction_per_unit' in cr and isinstance(cr.get('reduction_per_unit'), int):
        return cr['reduction_per_unit'] * applied

    if 'amount' in cr and isinstance(cr.get('amount'), int):
        return cr['amount']

    if 'unit_cost' in cr and isinstance(cr.get('unit_cost'), int):
        return cr['unit_cost'] * applied

    return 0


def apply_passive_reductions(card: Dict[str, Any], base_cost: int, units: int = 1) -> int:
    """Return adjusted cost after applying PASSIVE cost_reductions on the card.

    This is a prototype heuristic used for editor/agent-side pre-checks and tests.
    It does not replace engine-side semantics; final rules live in the engine.
    """
    if not isinstance(card, dict):
        return base_cost

    # Merge explicit cost_reductions with static COST_MODIFIERs for conservative evaluation.
    crs = _merged_passive_definitions(card)
    total_reduction = 0
    floor_min = None
    for cr in crs:
        if not isinstance(cr, dict):
            continue
        if cr.get('type') != 'PASSIVE':
            continue
        total_reduction += _compute_cr_reduction(cr, units)
        # track min_mana_cost if provided: we'll apply as floor
        if 'min_mana_cost' in cr and isinstance(cr.get('min_mana_cost'), int):
            if floor_min is None:
                floor_min = cr.get('min_mana_cost')
            else:
                # conservatively take the max floor among passives
                floor_min = max(floor_min, cr.get('min_mana_cost'))

    adjusted = max(base_cost - total_reduction, 0)
    if floor_min is not None:
        adjusted = max(adjusted, floor_min)
    return adjusted


def _sum_mana_pool(mana_pool: Any) -> int:
    if isinstance(mana_pool, dict):
        try:
            return sum(int(v) for v in mana_pool.values())
        except Exception:
            return 0
    try:
        return int(mana_pool)
    except Exception:
        return 0


def _has_required_civilization(card: Dict[str, Any], mana_pool: Any) -> bool:
    """Return True if mana_pool contains at least one mana of a civilization required by card.

    mana_pool may be an int (no per-civilization info) or a dict mapping civ->{count}.
    We conservatively require at least one mana of any declared civilization if the card
    specifies `civilization` (str) or `civilizations` (list).
    """
    if not isinstance(card, dict):
        return True
    req = None
    if 'civilization' in card and card.get('civilization'):
        req = [card.get('civilization')]
    elif 'civilizations' in card and card.get('civilizations'):
        try:
            req = list(card.get('civilizations'))
        except Exception:
            req = None

    if not req:
        return True

    if not isinstance(mana_pool, dict):
        # No per-civ info available; be conservative and assume it's insufficient
        return False

    # Check if any required civ has at least one mana in the pool
    for civ in req:
        if civ in mana_pool and int(mana_pool.get(civ, 0)) > 0:
            return True
    return False


def can_pay_with_mana(card: Dict[str, Any], mana_available: Any, base_cost: int, units: int = 1) -> bool:
    """Return True if mana_available can pay adjusted cost considering PASSIVE reductions.

    `mana_available` may be an integer (total mana) or a dict mapping civilization->count.
    If the card declares a required civilization, at least one mana of that civilization must
    be present in the mana pool dict form. If only an int is provided and the card requires
    a civilization, we conservatively return False.
    """
    plan = evaluate_cost(card, base_cost, units=units)
    total = _sum_mana_pool(mana_available)
    if total < plan.final_cost:
        return False
    return _has_required_civilization(card, mana_available)


def evaluate_cost(card: Dict[str, Any], base_cost: int, units: int = 1, active_reduction_id: Optional[str] = None, active_units: Optional[int] = None) -> PaymentPlan:
    """Produce a PaymentPlan describing passive and optional active reductions.

    This is a Python-side prototype to help design the engine `PaymentPlan`.
    """
    plan = PaymentPlan(
        base_cost=base_cost,
        adjusted_after_passive=base_cost,
        total_passive_reduction=0,
        passive_ids=[],
        active_reduction_id=active_reduction_id,
        active_units=active_units,
        active_reduction_amount=0,
        final_cost=base_cost,
    )

    # Apply PASSIVE reductions
    # Use merged passive definitions so that static COST_MODIFIER entries are
    # considered alongside explicit PASSIVE cost_reductions in editor/agent evaluation.
    crs = _merged_passive_definitions(card)
    floor_min = None
    total_red = 0
    passive_ids = []
    for cr in crs:
        if not isinstance(cr, dict):
            continue
        if cr.get('type') != 'PASSIVE':
            continue
        red = _compute_cr_reduction(cr, units)
        if red:
            total_red += red
            if 'id' in cr and cr.get('id'):
                passive_ids.append(cr.get('id'))
        if 'min_mana_cost' in cr and isinstance(cr.get('min_mana_cost'), int):
            if floor_min is None:
                floor_min = cr.get('min_mana_cost')
            else:
                floor_min = max(floor_min, cr.get('min_mana_cost'))

    adjusted = max(base_cost - total_red, 0)
    if floor_min is not None:
        adjusted = max(adjusted, floor_min)

    plan.adjusted_after_passive = adjusted
    plan.total_passive_reduction = total_red
    plan.passive_ids = passive_ids

    # Optionally apply ACTIVE_PAYMENT
    if active_reduction_id:
        red_amt = 0
        for cr in crs:
            if isinstance(cr, dict) and cr.get('id') == active_reduction_id and cr.get('type') == 'ACTIVE_PAYMENT':
                red_amt = _compute_active_reduction(cr, active_units or units)
                # respect min_mana_cost on the target active reduction
                floor = cr.get('min_mana_cost') if isinstance(cr.get('min_mana_cost'), int) else None
                break
        plan.active_reduction_amount = red_amt
        plan.final_cost = max(plan.adjusted_after_passive - red_amt, 0)
        if 'floor' in locals() and floor is not None:
            plan.final_cost = max(plan.final_cost, floor)
    else:
        plan.final_cost = plan.adjusted_after_passive

    return plan


def _compute_active_reduction(cr: Dict[str, Any], units: int) -> int:
    """Compute reduction for an ACTIVE_PAYMENT entry given selected units.

    Rules (prototype):
    - If `reduction_per_unit` present: reduction = reduction_per_unit * applied_units
    - applied_units = units if provided else cr.get('units') if present else 0
    - If `max_units` present, clamp applied_units to max_units
    - If `amount` present (one-shot), use that as reduction
    """
    if not isinstance(cr, dict):
        return 0
    try:
        applied = int(units) if units is not None else None
    except Exception:
        applied = None

    if applied is None:
        try:
            applied = int(cr.get('units')) if cr.get('units') is not None else 0
        except Exception:
            applied = 0

    try:
        max_units = int(cr.get('max_units')) if cr.get('max_units') is not None else 0
    except Exception:
        max_units = 0
    if max_units and max_units > 0:
        applied = min(applied, max_units)

    if 'reduction_per_unit' in cr and isinstance(cr.get('reduction_per_unit'), int):
        return cr['reduction_per_unit'] * applied

    if 'amount' in cr and isinstance(cr.get('amount'), int):
        return cr['amount']

    return 0


def _merged_passive_definitions(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of cost_reduction-like dicts including PASSIVE entries and
    converted `static_abilities` entries of type `COST_MODIFIER`.

    This helper is used only in the toolkit/editor evaluation paths to provide a
    conservative, unified view of passive cost modifiers coming from multiple
    data sources. It intentionally does not mutate the original card dict.
    """
    merged: List[Dict[str, Any]] = []
    if not isinstance(card, dict):
        return merged

    # Start with explicit cost_reductions if present
    explicit = card.get('cost_reductions') or []
    for cr in explicit:
        if isinstance(cr, dict):
            merged.append(cr.copy())

    # Convert static_abilities COST_MODIFIER entries into PASSIVE-like reductions
    statics = card.get('static_abilities') or []
    for idx, s in enumerate(statics):
        if not isinstance(s, dict):
            continue
        if s.get('type') == 'COST_MODIFIER':
            # Heuristic conversion: COST_MODIFIER.value -> PASSIVE.amount
            val = s.get('value')
            try:
                val_int = int(val)
            except Exception:
                continue
            if val_int == 0:
                continue
            merged.append({
                'type': 'PASSIVE',
                'id': f'static-cost-mod-{idx}',
                'amount': val_int,
            })

    return merged


def apply_active_payment(card: Dict[str, Any], base_cost: int, reduction_id: str, units: int = 1) -> int:
    """Apply a selected ACTIVE_PAYMENT reduction (by id) to base_cost and return adjusted cost.

    This prototype only applies the single selected ACTIVE_PAYMENT and does not combine multiple.
    """
    if not isinstance(card, dict) or not reduction_id:
        return base_cost

    crs = card.get('cost_reductions') or []
    target = None
    for cr in crs:
        if isinstance(cr, dict) and cr.get('id') == reduction_id and cr.get('type') == 'ACTIVE_PAYMENT':
            target = cr
            break

    if target is None:
        return base_cost

    reduction = _compute_active_reduction(target, units)
    # respect min_mana_cost floor if present
    floor = None
    if 'min_mana_cost' in target and isinstance(target.get('min_mana_cost'), int):
        floor = target.get('min_mana_cost')

    adjusted = max(base_cost - reduction, 0)
    if floor is not None:
        adjusted = max(adjusted, floor)
    return adjusted


def can_pay_with_active(card: Dict[str, Any], mana_available: int, base_cost: int, reduction_id: str, units: int = 1) -> bool:
    plan = evaluate_cost(card, base_cost, units=units, active_reduction_id=reduction_id, active_units=units)
    total = _sum_mana_pool(mana_available)
    return total >= plan.final_cost
