p='dm_ai_module.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
count_try=0
count_except=0
for i,l in enumerate(lines[:1800]):
    if 'try:' in l:
        count_try+=1
    if 'except ' in l:
        count_except+=1
    if i+1==1773:
        print('upto 1773: tries',count_try,'excepts',count_except)
print('total tries',count_try,'total excepts',count_except)
