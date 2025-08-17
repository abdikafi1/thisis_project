#!/usr/bin/env python3
"""
Test PostgreSQL connection ONLY - no SQLite fallback
This ensures your app works with PostgreSQL backend
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Testing PostgreSQL connection ONLY...")
print("=" * 60)

# Check if DATABASE_URL is set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("❌ DATABASE_URL not found!")
    print("💡 Create a .env file with:")
    print("   DATABASE_URL=postgresql://neondb_owner:npg_LXrJzj85qwcC@ep-soft-truth-a14lkz4p-pooler.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require")
    exit(1)

print(f"✅ DATABASE_URL found: {database_url[:50]}...")

# Test PostgreSQL connection
try:
    import psycopg2
    print("✅ psycopg2 is available")
    
    # Try to connect
    print("🔄 Testing connection to Neon database...")
    conn = psycopg2.connect(database_url)
    print("✅ Successfully connected to PostgreSQL!")
    
    # Test a simple query
    with conn.cursor() as cursor:
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Database version: {version[0] if version else 'Unknown'}")
        
        # Test if Django tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'landing_%' 
            LIMIT 5;
        """)
        tables = cursor.fetchall()
        if tables:
            print(f"✅ Found Django tables: {[t[0] for t in tables]}")
        else:
            print("⚠️  No Django tables found - you may need to run migrations")
    
    conn.close()
    print("✅ Connection test passed!")
    
except ImportError:
    print("❌ psycopg2 is not installed")
    print("💡 Install it with: pip install psycopg2-binary")
    exit(1)
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("💡 Check your internet connection and database credentials")
    print("💡 Make sure your Neon database is active")
    exit(1)

print("=" * 60)
print("🎯 Next steps:")
print("1. Run: python manage.py check")
print("2. Run: python manage.py migrate")
print("3. Run: python manage.py runserver 8000")
print("4. Test forgot password functionality")
print("=" * 60)
print("✅ PostgreSQL connection successful! Your app will work with PostgreSQL backend only.")
