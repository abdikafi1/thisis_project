#!/usr/bin/env python3
"""
Script to run Django migrations locally against Render PostgreSQL database
This is a free way to fix the database table issues
"""

import os
import sys
import django
from pathlib import Path

def setup_django():
    """Set up Django with production settings"""
    print("🔧 Setting up Django with production settings...")
    
    # Add project directory to Python path
    project_dir = Path(__file__).parent
    sys.path.insert(0, str(project_dir))
    
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings_production')
    
    # Set up Django
    django.setup()
    print("✅ Django setup complete")

def check_database_connection():
    """Check if we can connect to the database"""
    print("\n🔍 Checking database connection...")
    
    try:
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

def run_migrations():
    """Run Django migrations"""
    print("\n🚀 Running Django migrations...")
    
    try:
        from django.core.management import execute_from_command_line
        
        # Run migrations
        execute_from_command_line(['manage.py', 'migrate'])
        print("✅ Migrations completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def check_tables():
    """Check if tables were created"""
    print("\n🔍 Checking database tables...")
    
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            # Check for key tables
            tables_to_check = [
                'landing_prediction',
                'landing_userprofile', 
                'landing_useractivity',
                'landing_systemsettings',
                'auth_user',
                'django_migrations'
            ]
            
            for table in tables_to_check:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, [table])
                exists = cursor.fetchone()[0]
                status = "✅" if exists else "❌"
                print(f"   {status} {table}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return False

def create_superuser():
    """Create a superuser account"""
    print("\n👤 Creating superuser account...")
    
    try:
        from django.core.management import execute_from_command_line
        
        # Create superuser (non-interactive)
        execute_from_command_line(['manage.py', 'createsuperuser', '--noinput', '--username', 'admin', '--email', 'admin@example.com'])
        print("✅ Superuser created successfully!")
        print("   Username: admin")
        print("   Password: admin (change this after first login)")
        return True
        
    except Exception as e:
        print(f"⚠️  Superuser creation failed: {e}")
        print("   You can create one manually later")
        return False

def main():
    """Main function"""
    print("🚀 PostgreSQL Migration Script (Free Solution)")
    print("=" * 60)
    
    # Check if DATABASE_URL is set
    database_url = os.environ.get('DATABASE_URL', '')
    if not database_url:
        print("❌ DATABASE_URL not set!")
        print("\n💡 Set it first:")
        print("   Windows PowerShell:")
        print("     $env:DATABASE_URL='postgresql://ae:GabpfWUndScU8vhX1gEFm9DU5ijYlqs1@dpg-d2ede4ali9vc73dtrj60-a.singapore-postgres.render.com:5432/fraud_guird'")
        print("\n   Windows Command Prompt:")
        print("     set DATABASE_URL=postgresql://ae:GabpfWUndScU8vhX1gEFm9DU5ijYlqs1@dpg-d2ede4ali9vc73dtrj60-a.singapore-postgres.render.com:5432/fraud_guird")
        return
    
    print(f"✅ DATABASE_URL is set")
    
    # Setup Django
    setup_django()
    
    # Check database connection
    if not check_database_connection():
        print("❌ Cannot proceed without database connection")
        return
    
    # Run migrations
    if not run_migrations():
        print("❌ Migrations failed")
        return
    
    # Check tables
    check_tables()
    
    # Create superuser
    create_superuser()
    
    print("\n" + "=" * 60)
    print("🎉 Migration completed successfully!")
    print("\n💡 Next steps:")
    print("1. Test your app locally: python manage.py runserver")
    print("2. Commit and push your changes")
    print("3. Render will automatically redeploy")
    print("4. Your app should work without errors!")

if __name__ == "__main__":
    main()
