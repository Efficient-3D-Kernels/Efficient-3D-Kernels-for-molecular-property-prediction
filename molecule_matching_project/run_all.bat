@echo off
echo ========================================
echo MOLECULE PROCESSING AND MATCHING PIPELINE
echo ========================================

echo.
echo Step 1: Processing SDF files...
python process_sdf.py

echo.
echo Step 2: Running WCS matching...
python run_matching.py

echo.
echo ========================================
echo ALL DONE! Check the outputs folder.
echo ========================================
pause