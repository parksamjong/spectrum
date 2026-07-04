@echo off
cd /d "%~dp0"
start "Spectrum Dashboard" /min "C:\Users\user\AppData\Roaming\Python\Python38\Scripts\uvicorn.exe" main:app --host 0.0.0.0 --port 8000
timeout /t 2 /nobreak >nul
start "" "http://localhost:8000"
