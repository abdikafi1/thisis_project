from django.urls import path
from .views import (
    landing_page, prediction_view, history_view, dashboard_view, 
    get_started_view, prediction_result_view, login_view, signup_view, logout_view
)
 
urlpatterns = [
    path('', landing_page, name='landing_page'),
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', logout_view, name='logout'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('predict/', prediction_view, name='predict'),
    path('history/', history_view, name='prediction_history'),
    path('get-started/', get_started_view, name='get_started'),
    path('prediction-result/', prediction_result_view, name='prediction_result'),
] 