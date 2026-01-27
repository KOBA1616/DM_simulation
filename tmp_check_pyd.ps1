$env:PYTHONPATH = 'C:\Users\ichirou\DM_simulation\bin\Release;' + $env:PYTHONPATH
$env:DM_AI_MODULE_NATIVE = 'C:\Users\ichirou\DM_simulation\bin\Release\dm_ai_module.cp312-win_amd64.pyd'
& .\.venv\Scripts\python.exe -c "import dm_ai_module,sys,os; print('FILE:', getattr(dm_ai_module,'__file__',None)); print('PATH0..5', sys.path[:6]); print('PWD', os.getcwd())" 
