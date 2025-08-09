from django.urls import path
from .views import (
    landing_page, prediction_view, history_view, dashboard_view, 
    get_started_view, prediction_result_view, login_view, signup_view, logout_view,
    user_reports_view, fraud_analytics_view,
    user_settings_view, dashboard_settings_view, notification_settings_view,
    export_pdf_report, export_csv_report, export_excel_report,
    user_profile_view, user_dashboard_view
)
from .admin_views import (
    admin_dashboard, admin_user_management, admin_edit_user, admin_delete_user,
    admin_system_settings, admin_edit_setting, admin_activity_logs, admin_reports,
    admin_predict, admin_prediction_result, admin_analytics
)
 
urlpatterns = [
    # Landing and Authentication
    path('', landing_page, name='landing_page'),
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', logout_view, name='logout'),
    
    # User Dashboard and Profile  
    path('dashboard/', dashboard_view, name='dashboard'),
    path('user-dashboard/', user_dashboard_view, name='user_dashboard'),
    path('profile/', user_profile_view, name='user_profile'),
    
    # Prediction and Analysis
    path('predict/', prediction_view, name='predict'),
    path('home/', history_view, name='home'),
    path('get-started/', get_started_view, name='get_started'),
    path('prediction-result/', prediction_result_view, name='prediction_result'),
    
    # Reports and Analytics
    path('reports/', user_reports_view, name='user_reports'),
    path('reports/export/pdf/', export_pdf_report, name='export_pdf'),
    path('reports/export/csv/', export_csv_report, name='export_csv'),
    path('reports/export/excel/', export_excel_report, name='export_excel'),
    path('analytics/', fraud_analytics_view, name='fraud_analytics'),
    
    # Settings
    path('settings/user/', user_settings_view, name='user_settings'),
    path('settings/dashboard/', dashboard_settings_view, name='dashboard_settings'),
    path('settings/notifications/', notification_settings_view, name='notification_settings'),
    
    # Admin Routes (using 'manage' to avoid conflict with Django admin)
    path('manage/dashboard/', admin_dashboard, name='admin_dashboard'),
    path('manage/analytics/', admin_analytics, name='admin_analytics'),
    path('manage/predict/', admin_predict, name='admin_predict'),
    path('manage/prediction-result/', admin_prediction_result, name='admin_prediction_result'),
    path('manage/users/', admin_user_management, name='admin_user_management'),
    path('manage/users/<int:user_id>/edit/', admin_edit_user, name='admin_edit_user'),
    path('manage/users/<int:user_id>/delete/', admin_delete_user, name='admin_delete_user'),
    path('manage/settings/', admin_system_settings, name='admin_system_settings'),
    path('manage/settings/<int:setting_id>/edit/', admin_edit_setting, name='admin_edit_setting'),
    path('manage/activity-logs/', admin_activity_logs, name='admin_activity_logs'),
    path('manage/reports/', admin_reports, name='admin_reports'),
] 