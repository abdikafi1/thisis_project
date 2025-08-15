# Access Control System - Fraud Detection Project

## Overview
This document explains the new access control system implemented in the fraud detection project.

## Access Control Rules

### 1. User Types and Access Levels

#### **Basic Users** (`user_level = 'basic'`)
- **Can access:** User dashboard, predictions, reports, profile
- **Cannot access:** Admin features, admin dashboard
- **Django fields:** `is_superuser = False`, `is_staff = False`

#### **Admin Users** (`is_superuser = True` OR `is_staff = True`)
- **Can access:** Admin dashboard, user management, verification management
- **Cannot access:** User features, user dashboard
- **Django fields:** Either `is_superuser = True` OR `is_staff = True`

### 2. Important Rules

#### **Rule 1: Mutual Exclusivity**
- `is_superuser` AND `user_level = 'admin'` CANNOT both be true
- `is_staff` AND `user_level = 'basic'` CANNOT both be true
- Only ONE admin field can be true per user

#### **Rule 2: Access Separation**
- **Basic users** → User features only
- **Admin users** → Admin features only
- **No cross-access** between user and admin sections

### 3. Implementation Details

#### **Decorators Updated:**
- `@admin_required` - Checks `is_superuser OR is_staff`
- `@admin_or_basic_required` - Same logic as admin_required
- `@verified_user_required` - Checks verification status

#### **Views Updated:**
- `user_dashboard_view` - Blocks admin users
- `dashboard_view` - Redirects based on user type
- `prediction_view` - Blocks unverified users
- `history_view` - Blocks unverified users

#### **Templates Updated:**
- Admin templates use `user.is_superuser or user.is_staff`
- User templates check user type before allowing access

### 4. User Examples

#### **Example 1: Superuser Admin**
```python
user.is_superuser = True
user.is_staff = True
profile.user_level = 'basic'  # Must be basic for superuser
```
**Access:** Admin features only

#### **Example 2: Staff Admin**
```python
user.is_superuser = False
user.is_staff = True
profile.user_level = 'basic'  # Must be basic for staff
```
**Access:** Admin features only

#### **Example 3: Basic User**
```python
user.is_superuser = False
user.is_staff = False
profile.user_level = 'basic'
```
**Access:** User features only

### 5. Security Features

#### **Verification System:**
- Unverified users see warning banner
- Cannot access predictions or history
- Must contact admin for verification

#### **Access Control:**
- Admin users redirected from user dashboard
- User users redirected from admin dashboard
- Clear separation of concerns

### 6. Database Schema

#### **UserProfile Model:**
```python
class UserProfile(models.Model):
    user = models.OneToOneField(User)
    user_level = models.CharField(choices=[('basic', 'Basic User')])
    is_verified = models.BooleanField(default=False)
    # ... other fields
```

#### **Django User Model:**
```python
# Fields used for access control:
# - is_superuser: Boolean
# - is_staff: Boolean
# - is_active: Boolean
```

### 7. Testing the System

#### **Test Admin Access:**
1. Login as admin user
2. Try to access `/user-dashboard/` → Should redirect to admin dashboard
3. Access `/manage/verification/` → Should work

#### **Test User Access:**
1. Login as basic user
2. Try to access `/manage/dashboard/` → Should redirect to user dashboard
3. Access `/user-dashboard/` → Should work

### 8. Maintenance

#### **Adding New Admin Users:**
```python
# Option 1: Superuser
user.is_superuser = True
user.is_staff = True
profile.user_level = 'basic'

# Option 2: Staff Admin
user.is_superuser = False
user.is_staff = True
profile.user_level = 'basic'
```

#### **Adding New Basic Users:**
```python
user.is_superuser = False
user.is_staff = False
profile.user_level = 'basic'
```

## Summary
This system ensures:
- **Clear separation** between user and admin features
- **No privilege escalation** through multiple admin fields
- **Secure access control** with proper redirects
- **Maintainable code** with consistent logic
- **User verification** for additional security
