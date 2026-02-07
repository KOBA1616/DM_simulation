@echo off
echo ============================================================
echo C:\temp Directory Check
echo ============================================================
python check_temp_files.py
echo.
echo ============================================================
echo Running debug_draw_card.py
echo ============================================================
python debug_draw_card.py
echo.
echo ============================================================
echo Checking for temp files again
echo ============================================================
python check_temp_files.py
