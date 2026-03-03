@echo off
echo ========================================
echo  GramBrain AI - Starting Servers
echo ========================================
echo.

echo Starting Backend Server...
start "GramBrain Backend" cmd /k "cd backend && python main.py"

timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "GramBrain Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo  Servers Starting...
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to close this window...
pause >nul
