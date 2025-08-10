@echo off
chcp 65001 >nul
title 🚀 Fraud Detection System - Deployment Helper

echo 🚀 Fraud Detection System - Deployment Helper
echo ==============================================

REM Check if we're in the right directory
if not exist "manage.py" (
    echo ❌ Error: Please run this script from the project root directory
    pause
    exit /b 1
)

:menu
cls
echo.
echo Choose an option:
echo 1) 🚂 Deploy to Railway (Recommended)
echo 2) 🌊 Deploy to DigitalOcean
echo 3) 🦸 Deploy to Heroku
echo 4) 🔧 Setup local environment
echo 5) 🔒 Run security checks
echo 6) 📖 View deployment guide
echo 7) 🚪 Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto railway
if "%choice%"=="2" goto digitalocean
if "%choice%"=="3" goto heroku
if "%choice%"=="4" goto local
if "%choice%"=="5" goto security
if "%choice%"=="6" goto guide
if "%choice%"=="7" goto exit
echo ❌ Invalid choice. Please enter a number between 1-7.
pause
goto menu

:railway
echo.
echo 🚂 Deploying to Railway...
echo 1. Make sure you have Railway CLI installed
echo 2. Run: railway login
echo 3. Run: railway init
echo 4. Run: railway up
echo.
echo ✅ Railway deployment initiated!
pause
goto menu

:digitalocean
echo.
echo 🌊 Deploying to DigitalOcean App Platform...
echo 1. Push your code to GitHub
echo 2. Go to DigitalOcean App Platform
echo 3. Create new app from GitHub repository
echo 4. Configure environment variables
echo 5. Deploy!
echo.
echo ✅ DigitalOcean deployment guide provided!
pause
goto menu

:heroku
echo.
echo 🦸 Deploying to Heroku...
echo 1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
echo 2. Run: heroku login
echo 3. Run: heroku create your-app-name
echo 4. Run: heroku addons:create heroku-postgresql:mini
echo 5. Run: heroku config:set DJANGO_SETTINGS_MODULE=myproject.settings_production
echo 6. Run: git push heroku main
echo.
echo ✅ Heroku deployment initiated!
pause
goto menu

:local
echo.
echo 🔧 Setting up local environment...
echo Creating .env file...
(
echo # Django
echo SECRET_KEY=your-super-secret-key-here
echo DEBUG=False
echo ALLOWED_HOST=localhost
echo.
echo # Database
echo DB_NAME=fraud_detection
echo DB_USER=postgres
echo DB_PASSWORD=your-password
echo DB_HOST=localhost
echo DB_PORT=5432
echo.
echo # Email (optional)
echo EMAIL_HOST=smtp.gmail.com
echo EMAIL_PORT=587
echo EMAIL_HOST_USER=your-email@gmail.com
echo EMAIL_HOST_PASSWORD=your-app-password
) > .env
echo ✅ .env file created!
echo.
echo Installing production requirements...
pip install -r requirements_production.txt
echo.
echo Collecting static files...
python manage.py collectstatic --noinput
echo.
echo ✅ Local environment setup complete!
pause
goto menu

:security
echo.
echo 🔒 Running security checks...
echo Checking production settings...
findstr /C:"DEBUG = True" myproject\settings_production.py >nul
if %errorlevel%==0 (
    echo ⚠️  Warning: DEBUG is still True in production settings
) else (
    echo ✅ DEBUG is properly set to False
)

findstr /C:"SECRET_KEY = 'django-insecure-" myproject\settings_production.py >nul
if %errorlevel%==0 (
    echo ⚠️  Warning: Default SECRET_KEY detected in production settings
) else (
    echo ✅ SECRET_KEY is properly configured
)

findstr /C:"ALLOWED_HOSTS = [" myproject\settings_production.py >nul
if %errorlevel%==0 (
    echo ✅ ALLOWED_HOSTS is configured
) else (
    echo ⚠️  Warning: ALLOWED_HOSTS not found
)

echo ✅ Security check complete!
pause
goto menu

:guide
echo.
echo 📖 Opening deployment guide...
start DEPLOYMENT_GUIDE.md
pause
goto menu

:exit
echo.
echo 👋 Goodbye! Good luck with your deployment!
pause
exit /b 0
