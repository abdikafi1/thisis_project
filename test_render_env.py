#!/usr/bin/env python3
"""
Test script to simulate Render environment variables
This helps debug the DATABASE_URL issue
"""

import os
import sys

def test_render_environment():
    """Test environment variables as they would appear on Render"""
    print("🔍 Testing Render Environment Variables...")
    print("=" * 60)
    
    # Check if we're in a Render-like environment
    render_indicators = ['RENDER', 'PORT', 'DATABASE_URL']
    
    print("Environment Variables Check:")
    print("-" * 40)
    
    for indicator in render_indicators:
        value = os.environ.get(indicator, 'NOT SET')
        print(f"{indicator}: {repr(value)}")
    
    # Check for database-related variables
    print("\nDatabase-related Environment Variables:")
    print("-" * 40)
    db_vars = {}
    for key, value in os.environ.items():
        if any(db_key in key.upper() for db_key in ['DATABASE', 'DB', 'POSTGRES', 'PSQL']):
            db_vars[key] = value
            print(f"{key}: {repr(value)}")
    
    if not db_vars:
        print("❌ No database-related environment variables found")
    
    # Check if DATABASE_URL is set
    database_url = os.environ.get('DATABASE_URL', '')
    if database_url:
        print(f"\n✅ DATABASE_URL is set: {database_url[:50]}...")
        
        # Test if it's a valid PostgreSQL URL
        if 'postgres' in database_url.lower():
            print("✅ Looks like a valid PostgreSQL URL")
        else:
            print("⚠️  Doesn't look like a PostgreSQL URL")
    else:
        print("\n❌ DATABASE_URL is NOT set")
        
        # Check if we can find it in other variables
        print("\nSearching for potential database URLs in other variables...")
        for key, value in os.environ.items():
            if value and ('postgres' in value.lower() or '://' in value):
                print(f"  {key}: {value[:100]}...")
    
    return database_url

def simulate_render_database_url():
    """Simulate what the DATABASE_URL should look like on Render"""
    print("\n🔍 Simulating Render DATABASE_URL...")
    print("=" * 60)
    
    # This is what a typical Render PostgreSQL URL looks like
    sample_url = "postgresql://username:password@host:port/database_name"
    
    print("Expected DATABASE_URL format:")
    print(f"  {sample_url}")
    
    print("\nYour current DATABASE_URL:")
    current_url = os.environ.get('DATABASE_URL', 'NOT SET')
    print(f"  {current_url}")
    
    if current_url != 'NOT SET':
        print("\nURL Analysis:")
        if '://' in current_url:
            parts = current_url.split('://')
            if len(parts) == 2:
                scheme, rest = parts
                print(f"  Scheme: {scheme}")
                if '@' in rest:
                    auth, host_part = rest.split('@')
                    if ':' in auth:
                        user, password = auth.split(':')
                        print(f"  User: {user}")
                        print(f"  Password: {'*' * len(password)}")
                    else:
                        print(f"  User: {auth}")
                    
                    if ':' in host_part:
                        host_port, db_name = host_part.split('/')
                        if ':' in host_port:
                            host, port = host_port.split(':')
                            print(f"  Host: {host}")
                            print(f"  Port: {port}")
                        else:
                            print(f"  Host: {host_port}")
                        print(f"  Database: {db_name}")
                    else:
                        print(f"  Host: {host_part}")
                else:
                    print(f"  Connection string: {rest}")
        else:
            print("  ❌ Invalid URL format - missing '://'")

def main():
    """Main test function"""
    print("🚀 Render Environment Test Script")
    print("=" * 60)
    
    # Test current environment
    database_url = test_render_environment()
    
    # Simulate expected format
    simulate_render_database_url()
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    
    if database_url and database_url != 'NOT SET':
        print("✅ DATABASE_URL is configured")
        if 'postgres' in database_url.lower():
            print("✅ Looks like a valid PostgreSQL URL")
        else:
            print("⚠️  URL format might be incorrect")
    else:
        print("❌ DATABASE_URL is missing")
        print("💡 This is why your app is falling back to SQLite")
    
    print("\n💡 To fix on Render:")
    print("1. Check if PostgreSQL service 'fraud_guird' is running")
    print("2. Verify the service is linked to your web service")
    print("3. Check the environment variables in your web service")
    print("4. Ensure DATABASE_URL is automatically set from the database")

if __name__ == "__main__":
    main()
