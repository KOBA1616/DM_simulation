from pathlib import Path
p=Path(r'C:/Users/ichirou/DM_simulation/dm_toolkit/gui/editor/forms/command_form.py')
s=p.read_text(encoding='utf-8')
lines=s.splitlines()
new=[]
for L in lines:
    # preserve empty lines and comments
    if L.strip()=='' or L.lstrip().startswith('#'):
        new.append(L)
        continue
    # convert leading tabs to spaces
    leading=''
    i=0
    while i<len(L) and L[i] in (' ', '\t'):
        leading+=L[i]
        i+=1
    if leading:
        # count spaces and tabs
        spaces=leading.count(' ')
        tabs=leading.count('\t')
        total_spaces = spaces + tabs*4
        new_indent=' '*( (total_spaces//4)*4 )
    else:
        new_indent=''
    new.append(new_indent+L[i:])

p.write_text('\n'.join(new)+('\n' if s.endswith('\n') else ''), encoding='utf-8')
print('normalized', p)
