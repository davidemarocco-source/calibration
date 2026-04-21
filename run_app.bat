@echo off
REM Batch file to run the IRT Calibration Application

echo ========================================
echo IRT Parameter Calibration Application
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.

REM Install/upgrade requirements
if exist "requirements.txt" (
    echo Installing/updating required packages...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo Requirements installed successfully!
    echo.
) else (
    echo WARNING: requirements.txt not found
    echo.
)

REM Run the Streamlit app
echo Starting IRT Calibration Application...
echo.
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the server when done.
echo.
streamlit run irt_calibration_app.py

REM Deactivate virtual environment on exit
deactivate
