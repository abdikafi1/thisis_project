#!/usr/bin/env python
"""
Reset Admin Password Script
This script resets the admin user password for local development
"""

import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from django.contrib.auth.models import User

def reset_admin_password():
    try:
        # Get the admin user
        admin_user = User.objects.get(username='admin')
        
        # Reset password to Admin123!
        admin_user.set_password('Admin123!')
        admin_user.save()
        
        print("✅ Admin password reset successfully!")
        print("📝 Login credentials:")
        print("   Username: admin")
        print("   Password: Admin123!")
        print("\n🔗 You can now login at: http://127.0.0.1:8000/login/")
        
    except User.DoesNotExist:
        print("❌ Admin user not found. Creating new admin user...")
        
        # Create new admin user
        admin_user = User.objects.create_superuser(
            username='admin',
            email='admin@local.com',
            password='Admin123!'
        )
        
        print("✅ New admin user created successfully!")
        print("📝 Login credentials:")
        print("   Username: admin")
        print("   Password: Admin123!")
        print("\n🔗 You can now login at: http://127.0.0.1:8000/login/")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    reset_admin_password()
