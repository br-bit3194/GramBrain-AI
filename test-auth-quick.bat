@echo off
echo ========================================
echo Testing Authentication Integration
echo ========================================
echo.

echo Step 1: Installing backend dependencies...
cd backend
pip install -r requirements.txt
cd ..
echo.

echo Step 2: Testing authentication flow...
node test-auth-flow.js
echo.

echo ========================================
echo Test Complete!
echo ========================================
pause
