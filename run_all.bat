@echo off
REM One-click setup + run for Windows. Double-click it, or run it from CMD in this folder.
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv || goto :error
)
call venv\Scripts\activate || goto :error

echo Installing dependencies...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || goto :error

if not exist .env copy .env.example .env >nul

echo.
echo Running the data pipeline...
python -m bigdata.run_pipeline --engine spark --mode synthetic || goto :error

echo.
echo Verifying the results...
python verify_results.py || goto :error

echo.
echo Starting the dashboard at http://127.0.0.1:5000  (press Ctrl+C to stop)
python app.py
goto :eof

:error
echo.
echo Something failed. Read the message above, then see the Troubleshooting section of README.md
pause
