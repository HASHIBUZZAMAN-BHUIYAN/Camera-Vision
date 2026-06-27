@echo off
setlocal

set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Make sure Python 3.11+ is installed.
        pause
        exit /b 1
    )
)

echo Activating virtual environment and installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip
pip install -r "%~dp0requirements.txt"

if errorlevel 1 (
    echo ERROR: Package installation failed.
    pause
    exit /b 1
)

echo.
echo Setup complete! Virtual environment is ready.
echo.
echo To run the projects:
echo   Neural Network Visualization:
echo     cd "Neural Network Visualization"
echo     python app.py
echo.
echo   Vision Puzzle:
echo     cd "Vision puzzle"
echo     python app.py
echo.
echo The virtual environment stays active in this terminal session.
