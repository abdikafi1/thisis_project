#!/usr/bin/env python3
"""
Test script to verify database connection and environment variables
Run this locally to test your database setup before deploying
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_environment_variables():
    """Test if environment variables are set correctly"""
    print("🔍 Testing Environment Variables...")
    print("=" * 50)
    
    # Check for DATABASE_URL
    database_url = os.environ.get('DATABASE_URL', '')
    print(f"DATABASE_URL: {repr(database_url)}")
    
    if database_url:
        print("✅ DATABASE_URL is set")
        
        # Check if it looks like a PostgreSQL URL
        if 'postgres' in database_url.lower():
            print("✅ Looks like a PostgreSQL URL")
        else:
            print("⚠️  Doesn't look like a PostgreSQL URL")
    else:
        print("❌ DATABASE_URL is not set")
    
    # Check other important variables
    secret_key = os.environ.get('SECRET_KEY', '')
    debug = os.environ.get('DEBUG', '')
    allowed_hosts = os.environ.get('ALLOWED_HOSTS', '')
    
    print(f"\nSECRET_KEY: {'✅ Set' if secret_key else '❌ Not set'}")
    print(f"DEBUG: {debug}")
    print(f"ALLOWED_HOSTS: {allowed_hosts}")
    
    return database_url

def test_django_database_config():
    """Test Django database configuration"""
    print("\n🔍 Testing Django Database Configuration...")
    print("=" * 50)
    
    try:
        # Set Django settings module
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings_production')
        
        # Import Django and configure
        import django
        django.setup()
        
        # Test database connection
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"✅ Database connection successful!")
            print(f"   Database: {connection.settings_dict['NAME']}")
            print(f"   Engine: {connection.settings_dict['ENGINE']}")
            print(f"   Version: {version[0] if version else 'Unknown'}")
            
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_requirements():
    """Test if required packages are installed"""
    print("\n🔍 Testing Required Packages...")
    print("=" * 50)
    
    required_packages = [
        'django',
        'psycopg2',
        'dj_database_url',
        'gunicorn',
        'whitenoise'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
    
    return True

def main():
    """Main test function"""
    print("🚀 Database Connection Test Script")
    print("=" * 50)
    
    # Test 1: Environment variables
    database_url = test_environment_variables()
    
    # Test 2: Required packages
    test_requirements()
    
    # Test 3: Django database (only if DATABASE_URL is set)
    if database_url:
        test_django_database_config()
    else:
        print("\n⚠️  Skipping Django database test - no DATABASE_URL set")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    
    if database_url:
        print("✅ DATABASE_URL is configured")
        print("✅ Ready to test database connection")
    else:
        print("❌ DATABASE_URL is missing")
        print("💡 Set DATABASE_URL environment variable to test database")
    
    print("\n💡 To set DATABASE_URL locally:")
    print("   Windows: set DATABASE_URL=your_postgres_url")
    print("   Linux/Mac: export DATABASE_URL=your_postgres_url")
    print("   Or create a .env file with DATABASE_URL=your_postgres_url")

if __name__ == "__main__":
    main()
