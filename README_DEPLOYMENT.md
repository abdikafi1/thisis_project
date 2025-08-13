# Fraud Detection System - Deployment Guide

## Timezone Configuration

The system is configured to use **East Africa Time (EAT)** which is `Africa/Nairobi` timezone. This covers:
- Kenya (UTC+3)
- Uganda (UTC+3) 
- Tanzania (UTC+3)
- Ethiopia (UTC+3)
- Somalia (UTC+3)
- And other East African countries

### Timezone Settings

In `myproject/settings.py`:
```python
TIME_ZONE = 'Africa/Nairobi'
USE_TZ = True
```

### What This Affects

1. **Database Timestamps**: All `created_at`, `updated_at`, and other datetime fields in SQLite
2. **User Activity Logs**: Login times, prediction timestamps
3. **Reports**: Date/time displays in analytics and exports
4. **Dashboard**: All time-based calculations and displays

### Changing Timezone

If you need to use a different timezone, update the `TIME_ZONE` setting in `myproject/settings.py`:

```python
# Examples:
TIME_ZONE = 'UTC'                    # Universal Coordinated Time
TIME_ZONE = 'America/New_York'       # Eastern Time
TIME_ZONE = 'Europe/London'          # British Time
TIME_ZONE = 'Asia/Tokyo'             # Japan Time
```

# 🚀 Quick Deployment Guide

## 🎯 **Your System is Complete!**

Your fraud detection system has been fully configured for production deployment with:

✅ **Production settings** (`myproject/settings_production.py`)  
✅ **Production requirements** (`requirements_production.txt`)  
✅ **Railway configuration** (`railway.json`)  
✅ **DigitalOcean configuration** (`app.yaml`)  
✅ **Deployment guide** (`DEPLOYMENT_GUIDE.md`)  
✅ **Windows deployment script** (`deploy.bat`)  

---

## 🚂 **Railway (RECOMMENDED) - Best for ML Models**

**Why Railway?**
- 🚀 **Faster builds** than Render
- 🧠 **Better ML model support** (CatBoost, LightGBM, XGBoost)
- 🗄️ **PostgreSQL included**
- 💰 **Cost-effective**: $5/month for 500 hours

**Quick Deploy:**
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Add PostgreSQL database
4. Set environment variables
5. Deploy!

---

## 🌊 **DigitalOcean App Platform**

**Why DigitalOcean?**
- 🏗️ **Reliable infrastructure**
- 🗄️ **Managed databases**
- 🔒 **Good security features**

**Cost:** $5/month + database (~$15/month total)

---

## 🦸 **Heroku (Alternative)**

**Why Heroku?**
- 🎯 **Easy deployment**
- 👥 **Large community**
- 📚 **Many tutorials**

**Cost:** $7/month for basic dyno

---

## 🔧 **Quick Setup Commands**

```bash
# Install production requirements
pip install -r requirements_production.txt

# Collect static files
python manage.py collectstatic --noinput

# Run security checks
deploy.bat  # On Windows
```

---

## 🚨 **Before You Deploy**

1. **Change SECRET_KEY** in production settings
2. **Set DEBUG=False**
3. **Configure database credentials**
4. **Test ML models locally**

---

## 📖 **Need Help?**

- **Run:** `deploy.bat` (Windows)
- **Read:** `DEPLOYMENT_GUIDE.md`
- **Railway:** Excellent Discord community
- **DigitalOcean:** Good documentation

---

**🎉 Your fraud detection system is ready for the world! Choose Railway for the best ML model performance.**
