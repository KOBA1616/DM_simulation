import os

missing_props = [
    ('power_attacker', 'POWER_ATTACKER'),
    ('neo', 'NEO'),
    ('meta_counter_play', 'META_COUNTER_PLAY'),
    ('shield_burn', 'SHIELD_BURN'),
    ('untap_in', 'UNTAP_IN'),
    ('unblockable', 'UNBLOCKABLE'),
    ('friend_burst', 'FRIEND_BURST'),
    ('ex_life', 'EX_LIFE'),
    ('mega_last_burst', 'MEGA_LAST_BURST'),
    ('at_start_of_turn', 'AT_START_OF_TURN'),
    ('at_end_of_turn', 'AT_END_OF_TURN'),
    ('at_block', 'AT_BLOCK')
]

filepath = "src/bindings/bindings.cpp"
with open(filepath, 'r') as f:
    lines = f.readlines()

# Find where CardKeywords properties end
insert_idx = -1
for i, line in enumerate(lines):
    if '.def_property("hyper_energy"' in line:
        insert_idx = i + 1
        break

if insert_idx != -1:
    # Remove semicolon from the previous line (hyper_energy)
    # Only remove the last semicolon
    prev_line = lines[insert_idx - 1]
    stripped = prev_line.rstrip()
    if stripped.endswith(';'):
        # Remove the last char
        lines[insert_idx - 1] = stripped[:-1] + '\n'

    new_lines = []
    for i, (prop, enum_name) in enumerate(missing_props):
        # Check if already exists
        exists = False
        for line in lines:
            if f'.def_property("{prop}"' in line:
                exists = True
                break
        if not exists:
            # Use .has and .set
            line = f'        .def_property("{prop}", [](const CardKeywords& k) {{ return k.has(Keyword::{enum_name}); }}, [](CardKeywords& k, bool v) {{ k.set(Keyword::{enum_name}, v); }})'

            # Add semicolon to the last item
            if i == len(missing_props) - 1:
                line += ';\n'
            else:
                line += '\n'

            new_lines.append(line)
            print(f"Adding binding for {prop}")

    lines[insert_idx:insert_idx] = new_lines

    with open(filepath, 'w') as f:
        f.writelines(lines)
else:
    print("Could not find insertion point")
