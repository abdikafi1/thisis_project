from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import UserProfile, Prediction, UserActivity, SystemSettings

# Inline admin for UserProfile
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fields = ('user_level', 'phone_number', 'company', 'position', 'is_verified')

# Extend UserAdmin to include profile
class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'get_user_level', 'get_is_verified', 'is_active', 'date_joined')
    list_filter = ('is_active', 'profile__user_level', 'profile__is_verified', 'date_joined')
    search_fields = ('username', 'email', 'first_name', 'last_name', 'profile__company')
    
    def get_user_level(self, obj):
        return obj.profile.get_user_level_display() if hasattr(obj, 'profile') else 'No Profile'
    get_user_level.short_description = 'User Level'
    get_user_level.admin_order_field = 'profile__user_level'
    
    def get_is_verified(self, obj):
        return obj.profile.is_verified if hasattr(obj, 'profile') else False
    get_is_verified.short_description = 'Verified'
    get_is_verified.boolean = True
    get_is_verified.admin_order_field = 'profile__is_verified'

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'user_level', 'company', 'position', 'is_verified', 'created_at')
    list_filter = ('user_level', 'is_verified', 'created_at')
    search_fields = ('user__username', 'user__email', 'company', 'position')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Profile Details', {
            'fields': ('user_level', 'phone_number', 'company', 'position')
        }),
        ('Status', {
            'fields': ('is_verified',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'result', 'confidence_score', 'processing_time', 'created_at')
    list_filter = ('result', 'created_at')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('created_at', 'input_dict_display')
    date_hierarchy = 'created_at'
    
    def input_dict_display(self, obj):
        return obj.input_dict()
    input_dict_display.short_description = 'Input Data'

@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ('user', 'activity_type', 'description', 'ip_address', 'created_at')
    list_filter = ('activity_type', 'created_at')
    search_fields = ('user__username', 'user__email', 'description', 'ip_address')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Activity Information', {
            'fields': ('user', 'activity_type', 'description')
        }),
        ('Technical Details', {
            'fields': ('ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        }),
    )

@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    list_display = ('key', 'value', 'description', 'updated_at')
    search_fields = ('key', 'description')
    readonly_fields = ('updated_at',)
    
    fieldsets = (
        ('Setting Information', {
            'fields': ('key', 'value', 'description')
        }),
        ('Timestamp', {
            'fields': ('updated_at',)
        }),
    )

# Customize admin site headers
admin.site.site_header = 'Fraud Detection System Admin'
admin.site.site_title = 'Fraud Detection Admin'
admin.site.index_title = 'Welcome to Fraud Detection System Administration'
