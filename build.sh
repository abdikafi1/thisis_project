#!/usr/bin/env bash
# Build script for Render deployment

echo "Starting build process..."

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Run database migrations
python manage.py migrate

# Import existing data if available
if [ -f "data_backup.json" ]; then
    echo "Importing existing data..."
    python manage.py loaddata data_backup.json
    echo "Data import completed!"
else
    echo "No data backup found, starting with fresh database"
fi

# Create superuser if it doesn't exist
echo "Creating superuser..."
python manage.py create_superuser --username admin --email admin@frauddetection.com --password Admin123!

echo "Build completed successfully!"
