@echo off
echo ========================================
echo  Movie Genre Classification - CodSoft
echo  Intern: Vishal Kadwe
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/4] Python found!
python --version
echo.

REM Install dependencies
echo [2/4] Installing required libraries (this may take 1-2 minutes)...
pip install pandas numpy matplotlib seaborn scikit-learn -q
if errorlevel 1 (
    echo [WARNING] Some packages may already be installed. Continuing...
)
echo.

REM Run the script
echo [3/4] Running Movie Genre Classification...
echo ========================================
python movie_genre_task1.py
echo.

REM Check if files were created
echo [4/4] Checking output files...
if exist "genre_distribution.png" (
    echo [OK] Charts generated successfully!
) else (
    echo [WARNING] Some files may not have been created.
)

echo.
echo ========================================
echo  DONE! Check the folder for outputs.
echo ========================================
pause
