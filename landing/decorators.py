from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from functools import wraps
from django.http import HttpResponseForbidden

def admin_required(view_func):
    """Decorator to require admin level access or superuser"""
    # SUPERUSER PRIORITY SYSTEM:
    # 1. HIGHEST PRIORITY: request.user.is_superuser = True -> GRANT ACCESS IMMEDIATELY
    # 2. SECOND PRIORITY: profile.user_level = 'admin' -> GRANT ACCESS
    # 3. NEVER changes existing is_superuser values - respects database completely
    
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        # Check if user is Django superuser first
        if request.user.is_superuser:
            return view_func(request, *args, **kwargs)
        
        # Check if user has admin user_level from database
        from .models import UserProfile
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        if profile.user_level != 'admin':
            messages.error(request, 'Access denied. Admin privileges required.')
            return HttpResponseForbidden("Access denied. Admin privileges required.")
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def admin_or_basic_required(view_func):
    """Decorator to require admin or basic level access or superuser"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        # Check if user is Django superuser first
        if request.user.is_superuser:
            return view_func(request, *args, **kwargs)
        
        # Check if user has admin user_level from database
        from .models import UserProfile
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        if profile.user_level != 'admin':
            messages.error(request, 'This feature requires Admin access.')
            return HttpResponseForbidden("This feature requires Admin access.")
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def verified_user_required(view_func):
    """Decorator to require verified user status"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        from .models import UserProfile
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        if not profile.is_verified:
            messages.error(request, 'Your account needs to be verified to access this feature.')
            return HttpResponseForbidden("Account verification required.")
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def track_activity(activity_type, description_func=None):
    """Decorator to track user activities"""
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            # Execute the view function
            response = view_func(request, *args, **kwargs)
            
            # Track activity if user is authenticated
            if request.user.is_authenticated:
                from .models import UserActivity
                
                desc = description_func(request, *args, **kwargs) if description_func else f"User performed {activity_type}"
                
                UserActivity.objects.create(
                    user=request.user,
                    activity_type=activity_type,
                    description=desc,
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT', '')
                )
            
            return response
        return _wrapped_view
    return decorator
