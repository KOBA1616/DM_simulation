# -*- coding: utf-8 -*-
import inspect

from dm_toolkit.gui.editor.logic_tree import LogicTreeWidget


class _FakeIndex:
    def __init__(self, valid: bool = True) -> None:
        self._valid = valid

    def isValid(self) -> bool:
        return self._valid


class _FakeRawItem:
    def index(self) -> str:
        return "fake-index"


class _FakeQtEditorItem:
    def get_raw_item(self) -> _FakeRawItem:
        return _FakeRawItem()


class _FakeDataManager:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, str, dict[str, object]]] = []

    def apply_template_by_key(
        self,
        card_index: object,
        template_key: str,
        label: str,
        extra_context: dict[str, object] | None = None,
    ) -> _FakeQtEditorItem:
        self.calls.append((card_index, template_key, label, extra_context or {}))
        return _FakeQtEditorItem()


class _DummyWidget:
    def __init__(self) -> None:
        self.data_manager = _FakeDataManager()
        self.current_index: object | None = None
        self.expanded_index: object | None = None

    def setCurrentIndex(self, index: object) -> None:
        self.current_index = index

    def expand(self, index: object) -> None:
        self.expanded_index = index

    def _build_races_context(self, payload: dict[str, object] | None, context_key: str) -> dict[str, object]:
        return LogicTreeWidget._build_races_context(self, payload, context_key)


def test_template_methods_delegate_to_common_helper():
    methods = [
        LogicTreeWidget.add_rev_change,
        LogicTreeWidget.add_mekraid,
        LogicTreeWidget.add_friend_burst,
        LogicTreeWidget.add_mega_last_burst,
    ]
    for method in methods:
        src = inspect.getsource(method)
        assert "_apply_logic_template(" in src


def test_apply_logic_template_passes_races_payload_and_updates_focus():
    dummy = _DummyWidget()
    idx = _FakeIndex(valid=True)

    result = LogicTreeWidget._apply_logic_template(
        dummy,
        idx,
        template_key="MEKRAID",
        label="Mekraid",
        payload={"races": ["アーマード"]},
        races_context_key="mekraid_races",
    )

    assert isinstance(result, _FakeQtEditorItem)
    assert dummy.data_manager.calls == [
        (idx, "MEKRAID", "Mekraid", {"mekraid_races": ["アーマード"]})
    ]
    assert dummy.current_index == "fake-index"
    assert dummy.expanded_index is idx
