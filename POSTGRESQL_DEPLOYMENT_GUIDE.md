# PostgreSQL Migration & Render Deployment Guide

## Overview
This guide will help you migrate your fraud detection system from SQLite to PostgreSQL and deploy it on Render.

## What We've Set Up

### 1. Database Configuration
- **Development**: SQLite (local development)
- **Production**: PostgreSQL (Render deployment)
- Automatic database switching based on environment

### 2. New Files Created
- `myproject/settings_production.py` - Production settings
- `build.sh` - Build script for Render
- `render.yaml` - Render deployment configuration
- `migrate_to_postgres.py` - Data migration helper

### 3. Updated Files
- `myproject/settings.py` - Now supports both databases
- `requirements.txt` - Added PostgreSQL dependencies

## Step-by-Step Deployment

### Step 1: Install New Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test Local PostgreSQL (Optional)
If you want to test PostgreSQL locally first:

1. Install PostgreSQL on your machine
2. Create a database
3. Set environment variable:
   ```bash
   export DATABASE_URL="postgresql://username:password@localhost:5432/dbname"
   ```
4. Run migrations:
   ```bash
   python manage.py migrate
   ```

### Step 3: Deploy to Render

#### Option A: Using render.yaml (Recommended)
1. Push your code to GitHub
2. Connect your GitHub repo to Render
3. Render will automatically detect the `render.yaml` file
4. It will create both the web service and PostgreSQL database

#### Option B: Manual Setup
1. **Create PostgreSQL Database**:
   - Go to Render Dashboard
   - Create New → PostgreSQL
   - Choose Free plan
   - Note the connection string

2. **Create Web Service**:
   - Create New → Web Service
   - Connect your GitHub repo
   - Environment: Python
   - Build Command: `chmod +x build.sh && ./build.sh`
   - Start Command: `gunicorn myproject.wsgi:application`

3. **Set Environment Variables**:
   - `SECRET_KEY`: Generate a secure key
   - `DEBUG`: `false`
   - `DATABASE_URL`: Copy from PostgreSQL service
   - `ALLOWED_HOSTS`: `.onrender.com`

### Step 4: Database Migration
After deployment, your database will be empty. To migrate existing data:

1. **Option A**: Use the migration script
   ```bash
   python migrate_to_postgres.py
   ```

2. **Option B**: Manual data export/import
   ```bash
   # Export from SQLite
   python manage.py dumpdata > data_backup.json
   
   # Import to PostgreSQL (after setting DATABASE_URL)
   python manage.py loaddata data_backup.json
   ```

### Step 5: Verify Deployment
1. Check your Render service is running
2. Visit your app URL
3. Test the fraud detection functionality
4. Check admin panel: `/admin/`

## Environment Variables Reference

| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `SECRET_KEY` | Auto-generated | Render generated | Django secret key |
| `DEBUG` | `True` | `False` | Debug mode |
| `DATABASE_URL` | Empty (SQLite) | PostgreSQL URL | Database connection |
| `ALLOWED_HOSTS` | Local hosts | `.onrender.com` | Allowed domains |

## Database Models
Your system has these models that will be migrated:
- **UserProfile**: User account details and permissions
- **Prediction**: Fraud detection results and data
- **UserActivity**: User action logging
- **SystemSettings**: System configuration

## Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Check `DATABASE_URL` is set correctly
   - Ensure PostgreSQL service is running on Render

2. **Migration Errors**:
   - Run `python manage.py makemigrations` first
   - Then `python manage.py migrate`

3. **Static Files Not Loading**:
   - Check `build.sh` has execute permissions
   - Verify `STATIC_ROOT` is set correctly

4. **Import Errors**:
   - Ensure all requirements are installed
   - Check Python version compatibility

### Getting Help
- Check Render logs in the dashboard
- Verify environment variables are set
- Test locally with PostgreSQL first

## Benefits of PostgreSQL

1. **Production Ready**: Better for concurrent users
2. **Data Integrity**: ACID compliance
3. **Scalability**: Handles larger datasets
4. **Performance**: Better query optimization
5. **Render Integration**: Native support on Render

## Next Steps
After successful deployment:
1. Set up monitoring and logging
2. Configure custom domain (optional)
3. Set up automated backups
4. Monitor performance metrics

Your fraud detection system is now ready for production use with PostgreSQL on Render! 🚀

