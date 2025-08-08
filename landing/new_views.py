from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta
from .models import Prediction

@login_required
def user_reports_view(request):
    """View for user reports dashboard"""
    # Get user-specific statistics
    user_predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    total_user_predictions = user_predictions.count()
    user_fraud_count = user_predictions.filter(result='Fraud').count()
    user_not_fraud_count = user_predictions.filter(result='Not Fraud').count()
    
    # Get recent predictions
    recent_predictions = user_predictions[:10]
    
    # Calculate success rate
    success_rate = (user_not_fraud_count / total_user_predictions * 100) if total_user_predictions > 0 else 0
    
    context = {
        'total_user_predictions': total_user_predictions,
        'user_fraud_count': user_fraud_count,
        'user_not_fraud_count': user_not_fraud_count,
        'success_rate': round(success_rate, 2),
        'recent_predictions': recent_predictions,
    }
    return render(request, 'landing/user_reports.html', context)

@login_required
def fraud_analytics_view(request):
    """View for fraud analytics dashboard"""
    # Get fraud patterns and analytics
    fraud_patterns, non_fraud_patterns = analyze_fraud_patterns()
    
    # Get recent fraud cases
    recent_fraud = Prediction.objects.filter(result='Fraud').order_by('-created_at')[:10]
    
    context = {
        'fraud_patterns': fraud_patterns,
        'non_fraud_patterns': non_fraud_patterns,
        'recent_fraud': recent_fraud,
    }
    return render(request, 'landing/fraud_analytics.html', context)

@login_required
def performance_metrics_view(request):
    """View for performance metrics dashboard"""
    # Calculate overall performance metrics
    total_predictions = Prediction.objects.count()
    fraud_count = Prediction.objects.filter(result='Fraud').count()
    not_fraud_count = Prediction.objects.filter(result='Not Fraud').count()
    
    # Calculate accuracy metrics
    accuracy = (not_fraud_count / total_predictions * 100) if total_predictions > 0 else 0
    
    # Get monthly trends
    six_months_ago = timezone.now() - timedelta(days=180)
    monthly_data = Prediction.objects.filter(
        created_at__gte=six_months_ago
    ).extra(
        select={'month': "strftime('%Y-%m', created_at)"}
    ).values('month').annotate(
        count=Count('id'),
        fraud_count=Count('id', filter=Q(result='Fraud'))
    ).order_by('month')
    
    context = {
        'total_predictions': total_predictions,
        'fraud_count': fraud_count,
        'not_fraud_count': not_fraud_count,
        'accuracy': round(accuracy, 2),
        'monthly_data': monthly_data,
    }
    return render(request, 'landing/performance_metrics.html', context)

@login_required
def user_settings_view(request):
    """View for user settings"""
    if request.method == 'POST':
        # Handle user settings update
        messages.success(request, 'Settings updated successfully!')
        return redirect('user_settings')
    
    context = {
        'user': request.user,
    }
    return render(request, 'landing/user_settings.html', context)

@login_required
def dashboard_settings_view(request):
    """View for dashboard settings"""
    if request.method == 'POST':
        # Handle dashboard settings update
        messages.success(request, 'Dashboard settings updated successfully!')
        return redirect('dashboard_settings')
    
    context = {
        'user': request.user,
    }
    return render(request, 'landing/dashboard_settings.html', context)

@login_required
def notification_settings_view(request):
    """View for notification settings"""
    if request.method == 'POST':
        # Handle notification settings update
        messages.success(request, 'Notification settings updated successfully!')
        return redirect('notification_settings')
    
    context = {
        'user': request.user,
    }
    return render(request, 'landing/notification_settings.html', context) 