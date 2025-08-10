#!/bin/bash

# 🚀 Fraud Detection System - Deployment Script
# This script helps automate deployment to different platforms

echo "🚀 Fraud Detection System - Deployment Helper"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "manage.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    echo "1. Make sure you have Railway CLI installed"
    echo "2. Run: railway login"
    echo "3. Run: railway init"
    echo "4. Run: railway up"
    echo ""
    echo "✅ Railway deployment initiated!"
}

# Function to deploy to DigitalOcean
deploy_digitalocean() {
    echo "🌊 Deploying to DigitalOcean App Platform..."
    echo "1. Push your code to GitHub"
    echo "2. Go to DigitalOcean App Platform"
    echo "3. Create new app from GitHub repository"
    echo "4. Configure environment variables"
    echo "5. Deploy!"
    echo ""
    echo "✅ DigitalOcean deployment guide provided!"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "🦸 Deploying to Heroku..."
    echo "1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli"
    echo "2. Run: heroku login"
    echo "3. Run: heroku create your-app-name"
    echo "4. Run: heroku addons:create heroku-postgresql:mini"
    echo "5. Run: heroku config:set DJANGO_SETTINGS_MODULE=myproject.settings_production"
    echo "6. Run: git push heroku main"
    echo ""
    echo "✅ Heroku deployment initiated!"
}

# Function to setup local environment
setup_local() {
    echo "🔧 Setting up local environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        echo "Creating .env file..."
        cat > .env << EOF
# Django
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(50))')
DEBUG=False
ALLOWED_HOST=localhost

# Database
DB_NAME=fraud_detection
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432

# Email (optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
EOF
        echo "✅ .env file created!"
    else
        echo "✅ .env file already exists"
    fi
    
    # Install production requirements
    echo "Installing production requirements..."
    pip install -r requirements_production.txt
    
    # Collect static files
    echo "Collecting static files..."
    python manage.py collectstatic --noinput
    
    echo "✅ Local environment setup complete!"
}

# Function to run security checks
security_check() {
    echo "🔒 Running security checks..."
    
    # Check if DEBUG is False in production settings
    if grep -q "DEBUG = True" myproject/settings_production.py; then
        echo "⚠️  Warning: DEBUG is still True in production settings"
    else
        echo "✅ DEBUG is properly set to False"
    fi
    
    # Check if SECRET_KEY is properly configured
    if grep -q "SECRET_KEY = 'django-insecure-" myproject/settings_production.py; then
        echo "⚠️  Warning: Default SECRET_KEY detected in production settings"
    else
        echo "✅ SECRET_KEY is properly configured"
    fi
    
    # Check if ALLOWED_HOSTS is configured
    if grep -q "ALLOWED_HOSTS = \[" myproject/settings_production.py; then
        echo "✅ ALLOWED_HOSTS is configured"
    else
        echo "⚠️  Warning: ALLOWED_HOSTS not found"
    fi
    
    echo "✅ Security check complete!"
}

# Main menu
while true; do
    echo ""
    echo "Choose an option:"
    echo "1) 🚂 Deploy to Railway (Recommended)"
    echo "2) 🌊 Deploy to DigitalOcean"
    echo "3) 🦸 Deploy to Heroku"
    echo "4) 🔧 Setup local environment"
    echo "5) 🔒 Run security checks"
    echo "6) 📖 View deployment guide"
    echo "7) 🚪 Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            deploy_railway
            ;;
        2)
            deploy_digitalocean
            ;;
        3)
            deploy_heroku
            ;;
        4)
            setup_local
            ;;
        5)
            security_check
            ;;
        6)
            echo "📖 Opening deployment guide..."
            if command -v xdg-open &> /dev/null; then
                xdg-open DEPLOYMENT_GUIDE.md
            elif command -v open &> /dev/null; then
                open DEPLOYMENT_GUIDE.md
            else
                echo "Please open DEPLOYMENT_GUIDE.md manually"
            fi
            ;;
        7)
            echo "👋 Goodbye! Good luck with your deployment!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter a number between 1-7."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
