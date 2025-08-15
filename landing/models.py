from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver
import json

class UserProfile(models.Model):
    USER_LEVELS = [
        ('basic', 'Basic User'),
        ('admin', 'Administrator'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    user_level = models.CharField(max_length=10, choices=USER_LEVELS, default='basic')
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    company = models.CharField(max_length=100, blank=True, null=True)
    position = models.CharField(max_length=100, blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.get_user_level_display()}"
    
    @property
    def is_admin(self):
        return self.user_level == 'admin'
    
    @property
    def is_basic(self):
        return self.user_level == 'basic'
    
    def save(self, *args, **kwargs):
        """Enforce rule: only one admin type can be active per user"""
        if self.user_level == 'admin':
            # If user_level is admin, ensure Django admin fields are False
            self.user.is_superuser = False
            self.user.is_staff = False
            self.user.save(update_fields=['is_superuser', 'is_staff'])
        elif self.user.is_superuser or self.user.is_staff:
            # If Django admin fields are True, ensure user_level is basic
            self.user_level = 'basic'
        
        super().save(*args, **kwargs)
    
    @classmethod
    def set_user_as_admin(cls, user, admin_type='custom'):
        """
        Set user as admin with specified type
        admin_type: 'django' (uses Django admin) or 'custom' (uses custom admin)
        """
        if admin_type == 'django':
            # Django admin: set Django fields, ensure custom level is basic
            user.is_superuser = True
            user.is_staff = True
            user.save()
            profile, created = cls.objects.get_or_create(user=user)
            profile.user_level = 'basic'
            profile.save()
        else:
            # Custom admin: set custom level, ensure Django fields are False
            user.is_superuser = False
            user.is_staff = False
            user.save()
            profile, created = cls.objects.get_or_create(user=user)
            profile.user_level = 'admin'
            profile.save()
        
        return profile

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions', null=True, blank=True)
    input_data = models.TextField()  # Store as JSON string
    result = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']

    def input_dict(self):
        return json.loads(self.input_data)

class UserActivity(models.Model):
    ACTIVITY_TYPES = [
        ('login', 'User Login'),
        ('prediction', 'Made Prediction'),
        ('report_export', 'Exported Report'),
        ('settings_change', 'Changed Settings'),
        ('admin_action', 'Admin Action'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPES)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class SystemSettings(models.Model):
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.key}: {self.value}"

# Signal to create user profile when user is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save() 