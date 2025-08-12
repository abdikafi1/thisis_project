from django.db.models import Count, Q
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Prediction, UserActivity

def analytics_data(request):
    """Context processor to provide analytics data to all templates"""
    if not request.user.is_authenticated:
        return {}
    
    try:
        # Get user-specific analytics
        user_predictions = Prediction.objects.filter(user=request.user)
        total_user_predictions = user_predictions.count()
        user_fraud_predictions = user_predictions.filter(result='Fraud').count()
        user_safe_predictions = user_predictions.filter(result='Not Fraud').count()
        
        # Calculate user accuracy rate
        if total_user_predictions > 0:
            fraud_accuracy = (user_fraud_predictions / total_user_predictions) * 100 if user_fraud_predictions > 0 else 0
            safe_accuracy = (user_safe_predictions / total_user_predictions) * 100 if user_safe_predictions > 0 else 0
            user_accuracy_rate = round((fraud_accuracy + safe_accuracy) / 2, 1)
        else:
            user_accuracy_rate = 0
        
        # Get this month's predictions for user
        this_month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        this_month_predictions = user_predictions.filter(created_at__gte=this_month_start).count()
        
        # Get global analytics (for admin users)
        total_all_predictions = Prediction.objects.count()
        total_fraud_detected = Prediction.objects.filter(result='Fraud').count()
        
        # Calculate global detection accuracy
        if total_all_predictions > 0:
            fraud_rate = (total_fraud_detected / total_all_predictions) * 100
            global_detection_accuracy = round(100 - fraud_rate, 1)
        else:
            global_detection_accuracy = 0
        
        return {
            'sidebar_analytics': {
                'user_predictions': total_user_predictions,
                'fraud_predictions': user_fraud_predictions,
                'safe_predictions': user_safe_predictions,
                'accuracy_rate': user_accuracy_rate,
                'this_month_predictions': this_month_predictions,
                'total_all_predictions': total_all_predictions,
                'total_fraud_detected': total_fraud_detected,
                'detection_accuracy': global_detection_accuracy,
            }
        }
    except Exception as e:
        # Return empty data if there's an error
        return {
            'sidebar_analytics': {
                'user_predictions': 0,
                'fraud_predictions': 0,
                'safe_predictions': 0,
                'accuracy_rate': 0,
                'this_month_predictions': 0,
                'total_all_predictions': 0,
                'total_fraud_detected': 0,
                'detection_accuracy': 0,
            }
        }
