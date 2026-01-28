Add-Type -AssemblyName System.Management.Automation
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('C:\Users\ichirou\DM_simulation\scripts\run_gui.ps1',[ref]$tokens,[ref]$errors)
if ($errors) { $errors | Format-List } else { Write-Host 'PARSE_OK' }
