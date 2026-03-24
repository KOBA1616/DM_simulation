cd /d C:\Users\ichirou\DM_simulation
git switch -c feature/editor-stat-key-select || git switch feature/editor-stat-key-select
git add -A
git commit -m "editor: make STAT select; add tests; remove legacy condition config" || echo "No changes to commit or commit failed"
