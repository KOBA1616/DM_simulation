with open('build_check.txt', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

with open('build_result.txt', 'w', encoding='utf-8') as out:
    for l in lines:
        ls = l.strip()
        if ('FAILED' in ls or 'succeeded' in ls or 
            ('error C' in ls and 'warning' not in ls.lower()) or
            'Build succeeded' in ls):
            out.write(ls[:200] + '\n')
    # Also write last 30 lines  
    out.write('\n--- LAST 30 LINES ---\n')
    for l in lines[-30:]:
        out.write(l[:200])
