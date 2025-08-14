#!/bin/bash
set -e

echo "Starting build process..."
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Pip version: $(pip --version)"
echo "Current directory: $(pwd)"
echo "Files in current directory: $(ls -la)"

echo "Checking environment variables..."
echo "DATABASE_URL: $DATABASE_URL"
echo "DJANGO_SETTINGS_MODULE: $DJANGO_SETTINGS_MODULE"
echo "Running on Render: $RENDER"

echo "Installing requirements..."
pip install -r requirements_deploy.txt

echo "Verifying gunicorn installation..."
pip list | grep gunicorn || echo "Gunicorn not found in pip list"
which gunicorn || echo "Gunicorn not found in PATH"

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Running migrations..."
python manage.py migrate --verbosity=2

echo "Checking if tables exist..."
python manage.py dbshell -c "\dt" || echo "Cannot check tables, proceeding with syncdb"

echo "Forcing table creation with syncdb..."
python manage.py migrate --run-syncdb --verbosity=2

echo "Verifying tables were created..."
python manage.py dbshell -c "\dt" || echo "Table verification failed"

echo "Loading data..."
python manage.py loaddata data_backup.json || echo "Data import failed, continuing with fresh database"

echo "Build completed successfully!"
