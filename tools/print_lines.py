p='C:/Users/ichirou/DM_simulation/scripts/run_gui.ps1'
with open(p, encoding='utf-8') as f:
    for i,line in enumerate(f, start=1):
        print(f"{i:4}: {line.rstrip()}")
