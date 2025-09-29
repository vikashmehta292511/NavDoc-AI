@echo off
title NavDoc AI Setup
color 0A

echo ========================================
echo    NavDoc AI - Medical Document Analyzer
echo    Automated Setup for Windows
echo ========================================
echo.

echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo Python found!

echo.
echo [2/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
)

echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/6] Installing Python dependencies...
echo This may take 5-15 minutes depending on your internet speed...
pip install -r requirements.txt

echo.
echo [5/6] Checking Tesseract installation...
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo Tesseract found!
) else (
    echo WARNING: Tesseract OCR not found!
    echo Please install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo Or run tesseract installer.exe if available in this directory
    pause
)

echo.
echo [6/6] Setup complete!
echo.
echo Starting NavDoc AI server...
echo.
echo When server starts:
echo 1. Open your browser
echo 2. Go to http://127.0.0.1:8000
echo 3. Open index.html file
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn medical_ai_backend:app --reload

pause