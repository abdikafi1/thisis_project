#!/usr/bin/env bash
# Build script for Render deployment

echo "Starting build process..."

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Run database migrations
echo "Running database migrations..."
python manage.py migrate

# Import existing data if available
if [ -f "data_backup.json" ]; then
    echo "Importing existing data..."
    python manage.py loaddata data_backup.json || echo "Data import failed, continuing with fresh database"
    echo "Data import completed!"
else
    echo "No data backup found, starting with fresh database"
fi

echo "Build completed successfully!"
