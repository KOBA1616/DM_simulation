from dm_toolkit.gui.editor.forms.diff_tree_widget import DiffTreeWidget


def test_list_selection_and_toggle():
    w = DiffTreeWidget()
    tree = {'a': True, 'b': {'x': True}, 'options': {1: {'label': True}}}
    w.set_diff_tree(tree)
    lines = w.get_lines()
    assert 'a' in lines
    assert 'b.x' in lines
    assert 'options[1].label' in lines

    # enable selectable mode and select some lines
    w.set_selectable(True)
    w.select_lines(['b.x'])
    sel = w.get_selected_lines()
    assert 'b.x' in sel

    # toggle another line
    w.toggle_line_selected('options[1].label')
    sel2 = w.get_selected_lines()
    assert 'options[1].label' in sel2

    # deselect by toggling again
    w.toggle_line_selected('options[1].label')
    sel3 = w.get_selected_lines()
    assert 'options[1].label' not in sel3
