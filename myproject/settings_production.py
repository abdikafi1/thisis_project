"""
Production settings for fraud detection system on Render
"""
import os
from pathlib import Path

# Import specific settings from base settings to avoid conflicts
from .settings import (
    BASE_DIR, SECRET_KEY, INSTALLED_APPS, MIDDLEWARE, ROOT_URLCONF,
    TEMPLATES, WSGI_APPLICATION, AUTH_PASSWORD_VALIDATORS,
    LANGUAGE_CODE, TIME_ZONE, USE_I18N, USE_TZ, STATIC_URL,
    DEFAULT_AUTO_FIELD, LOGIN_URL, LOGIN_REDIRECT_URL
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Production hosts
ALLOWED_HOSTS = [
    '.onrender.com',
    'localhost',
    '127.0.0.1',
]

# Database configuration for PostgreSQL on Render
import dj_database_url

# Get DATABASE_URL and ensure it's a string
# Try multiple methods to get the DATABASE_URL
database_url = None

# Debug: Print all environment variables
print("All environment variables:")
for key, value in os.environ.items():
    if 'DATABASE' in key.upper() or 'DB' in key.upper():
        print(f"  {key}: {repr(value)}")

# Method 1: Try using decouple config
try:
    from decouple import config
    database_url = config('DATABASE_URL', default='')
    print(f"Got DATABASE_URL from decouple config: {repr(database_url)}")
except ImportError:
    print("decouple not available, trying os.environ")

# Method 2: Try os.environ directly
if not database_url:
    database_url = os.environ.get('DATABASE_URL', '')
    print(f"Got DATABASE_URL from os.environ: {repr(database_url)}")

# Method 3: Try os.getenv as fallback
if not database_url:
    database_url = os.getenv('DATABASE_URL', '')
    print(f"Got DATABASE_URL from os.getenv: {repr(database_url)}")

print(f"Final DATABASE_URL: {repr(database_url)}")

if database_url:
    # Handle bytes objects by decoding to string if necessary
    if isinstance(database_url, bytes):
        print(f"Converting bytes to string: {database_url}")
        database_url = database_url.decode('utf-8')
    
    # Clean the URL string (remove any extra quotes or whitespace)
    database_url = database_url.strip().strip('"').strip("'")
    print(f"Cleaned DATABASE_URL: {repr(database_url)}")
    
    try:
        # Override the database configuration from base settings
        DATABASES = {
            'default': dj_database_url.parse(database_url)
        }
        print(f"Successfully configured PostgreSQL database")
        print(f"Database engine: {DATABASES['default']['ENGINE']}")
        print(f"Database name: {DATABASES['default']['NAME']}")
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}")
        print(f"DATABASE_URL value: {repr(database_url)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        # Fallback to SQLite if parsing fails
        DATABASES = {
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': BASE_DIR / 'db.sqlite3',
            }
        }
        print("Using SQLite fallback due to parsing error")
else:
    print("No DATABASE_URL found, using SQLite fallback")
    # Fallback to default SQLite if no DATABASE_URL is provided
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

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
