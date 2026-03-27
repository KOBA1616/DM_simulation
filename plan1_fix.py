import re

with open("dm_toolkit/gui/editor/formatters/special_effect_formatters.py", "r") as f:
    se_content = f.read()

# Update AddKeywordFormatter
old_add_keyword_block = """        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)"""
new_add_keyword_block = """        max_cost_src = get_command_max_cost(command)
        val1 = max_cost_src if max_cost_src is not None else get_command_amount(command, default=0)"""
se_content = se_content.replace(old_add_keyword_block, new_add_keyword_block)

# Update MutateFormatter
old_mutate_block = """        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)"""
new_mutate_block = """        max_cost_src = get_command_max_cost(command)
        val1 = max_cost_src if max_cost_src is not None else get_command_amount(command, default=0)"""
se_content = se_content.replace(old_mutate_block, new_mutate_block)

# Update SummonTokenFormatter
old_summon_token_block = """        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)"""
new_summon_token_block = """        max_cost_src = get_command_max_cost(command)
        val1 = max_cost_src if max_cost_src is not None else get_command_amount(command, default=0)"""
se_content = se_content.replace(old_summon_token_block, new_summon_token_block)

# Update RegisterDelayedEffectFormatter
old_register_delayed_block = """        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)"""
new_register_delayed_block = """        max_cost_src = get_command_max_cost(command)
        val1 = max_cost_src if max_cost_src is not None else get_command_amount(command, default=0)"""
se_content = se_content.replace(old_register_delayed_block, new_register_delayed_block)

with open("dm_toolkit/gui/editor/formatters/special_effect_formatters.py", "w") as f:
    f.write(se_content)

print("special_effect_formatters.py updated.")
