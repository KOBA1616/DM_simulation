import os

filepath = "src/bindings/bindings.cpp"
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if '.def_readwrite("power", &CardDefinition::power)' in line:
        new_lines.append('        .def_readwrite("power_attacker_bonus", &CardDefinition::power_attacker_bonus)\n')

with open(filepath, 'w') as f:
    f.writelines(new_lines)
