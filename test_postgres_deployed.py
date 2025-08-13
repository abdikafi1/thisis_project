#!/usr/bin/env python
"""
Test PostgreSQL connection after deployment on Render
"""
import os
import django

def test_postgres_connection():
    """Test if PostgreSQL is working after deployment"""
    try:
        # Setup Django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
        django.setup()
        
        # Test database connection
        from django.db import connection
        cursor = connection.cursor()
        
        # Get database info
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"✅ Connected to: {db_version[0]}")
        
        # Test Django models
        from django.contrib.auth.models import User
        from landing.models import UserProfile, Prediction, UserActivity, SystemSettings
        
        # Count records
        user_count = User.objects.count()
        profile_count = UserProfile.objects.count()
        prediction_count = Prediction.objects.count()
        activity_count = UserActivity.objects.count()
        
        print(f"\n📊 Data verification:")
        print(f"  Users: {user_count}")
        print(f"  User Profiles: {profile_count}")
        print(f"  Predictions: {prediction_count}")
        print(f"  User Activities: {activity_count}")
        
        # Test if data was imported
        if user_count > 0:
            print(f"\n🎉 PostgreSQL migration successful!")
            print(f"Your fraud detection system is running on PostgreSQL!")
            return True
        else:
            print(f"\n⚠️  Database is empty - data may not have been imported")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Check:")
        print("1. Is your app deployed on Render?")
        print("2. Is DATABASE_URL set correctly?")
        print("3. Have migrations run?")
        return False

if __name__ == '__main__':
    print("🧪 Testing PostgreSQL connection after deployment...")
    print("=" * 50)
    
    if test_postgres_connection():
        print("\n✅ Test passed - PostgreSQL is working!")
    else:
        print("\n❌ Test failed - check deployment status")

