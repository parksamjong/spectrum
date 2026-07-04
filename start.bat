@echo off
cd /d "%~dp0"
start "Spectrum Dashboard" /min "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
timeout /t 3 /nobreak >nul
start "" "http://localhost:8000"
