"""
Production settings for fraud detection system on Render
"""
import os
from pathlib import Path

# Import specific settings from base settings to avoid conflicts
from .settings import (
    BASE_DIR, INSTALLED_APPS, MIDDLEWARE, ROOT_URLCONF,
    TEMPLATES, WSGI_APPLICATION, AUTH_PASSWORD_VALIDATORS,
    LANGUAGE_CODE, TIME_ZONE, USE_I18N, USE_TZ, STATIC_URL,
    DEFAULT_AUTO_FIELD, LOGIN_URL, LOGIN_REDIRECT_URL
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-w^=fb$xyf-ku5@n^*ek4#x(-iijee+kj$ixqxmw3wr0zjz@y(*')

# Production hosts
ALLOWED_HOSTS = [
    '.onrender.com',
    'localhost',
    '127.0.0.1',
]

# Database configuration for PostgreSQL on Render
import dj_database_url

# Get DATABASE_URL from environment
database_url = os.environ.get('DATABASE_URL', '')

# Debug: Print database-related environment variables
print("Checking for database-related environment variables:")
for key, value in os.environ.items():
    if any(db_key in key.upper() for db_key in ['DATABASE', 'DB', 'POSTGRES', 'PSQL']):
        print(f"  {key}: {repr(value)}")

print(f"DATABASE_URL from environment: {repr(database_url)}")

# Also check if we're running in Render environment
print(f"Running on Render: {'RENDER' in os.environ}")
print(f"All environment variables: {list(os.environ.keys())}")

if database_url:
    # Clean the URL string (remove any extra quotes or whitespace)
    database_url = database_url.strip().strip('"').strip("'")
    print(f"Cleaned DATABASE_URL: {repr(database_url)}")
    
    try:
        # Parse the database URL
        db_config = dj_database_url.parse(database_url)
        
        # Add SSL requirements for Render PostgreSQL
        db_config['OPTIONS'] = {
            'sslmode': 'require',
        }
        
        # Configure PostgreSQL database
        DATABASES = {
            'default': db_config
        }
        print(f"Successfully configured PostgreSQL database")
        print(f"Database engine: {DATABASES['default']['ENGINE']}")
        print(f"Database name: {DATABASES['default']['NAME']}")
        print(f"Database host: {DATABASES['default']['HOST']}")
        print(f"Database port: {DATABASES['default']['PORT']}")
        print(f"SSL mode: {DATABASES['default']['OPTIONS']['sslmode']}")
            
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}")
        print(f"DATABASE_URL value: {repr(database_url)}")
        raise Exception("Failed to configure PostgreSQL database. Please check your DATABASE_URL.")
                            else:
    print("No DATABASE_URL found!")
    raise Exception("DATABASE_URL environment variable is required for production deployment.")

# Security settings for production
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Static files configuration for production
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

# Cache settings
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    }
}

# Email settings (configure as needed)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
