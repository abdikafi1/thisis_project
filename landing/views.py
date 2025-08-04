from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import PredictionForm
import pandas as pd
import joblib
import os
from .models import Prediction
import json
from django.db.models import Q

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')
CATEGORICAL_COLS_PATH = os.path.join(MODEL_DIR, 'categorical_cols.pkl')

# Load model and encoders once
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
categorical_cols = joblib.load(CATEGORICAL_COLS_PATH)

def predict_fraud(input_data):
    input_df = pd.DataFrame([input_data])
    errors = []
    for col in categorical_cols:
        le = encoders[col]
        try:
            input_df[col] = le.transform(input_df[col].astype(str).str.strip())
        except ValueError:
            errors.append(f"Invalid value for {col}: {input_df[col].values[0]}. Please select a valid option.")
            input_df[col] = -1
    if errors:
        return None, errors
    pred = model.predict(input_df)[0]
    return pred, None

@login_required
def prediction_view(request):
    return redirect('dashboard')

@login_required
def history_view(request):
    # Get filter parameters
    filter_result = request.GET.get('filter_result', 'all')
    search_query = request.GET.get('search', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    # Start with all predictions
    history = Prediction.objects.all().order_by('-created_at')
    
    # Apply filters
    if filter_result == 'fraud':
        history = history.filter(result='Fraud')
    elif filter_result == 'not_fraud':
        history = history.filter(result='Not Fraud')
    
    # Apply search if provided
    if search_query:
        history = history.filter(
            Q(result__icontains=search_query) |
            Q(input_data__icontains=search_query)
        )
    
    # Apply date filters
    if date_from:
        history = history.filter(created_at__gte=date_from)
    if date_to:
        history = history.filter(created_at__lte=date_to)
    
    # Get statistics
    total_predictions = Prediction.objects.count()
    fraud_count = Prediction.objects.filter(result='Fraud').count()
    not_fraud_count = Prediction.objects.filter(result='Not Fraud').count()
    
    context = {
        'history': history,
        'filter_result': filter_result,
        'search_query': search_query,
        'date_from': date_from,
        'date_to': date_to,
        'total_predictions': total_predictions,
        'fraud_count': fraud_count,
        'not_fraud_count': not_fraud_count,
        'filtered_count': history.count(),
    }
    
    return render(request, 'landing/history.html', context)

@login_required
def dashboard_view(request):
    result = None
    errors = None
    filter_result = request.GET.get('filter_result', 'all')
    search_query = request.GET.get('search', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data
            pred, errors = predict_fraud(input_data)
            if errors:
                result = None
            else:
                result = 'Fraud' if pred == 1 else 'Not Fraud'
                # Save to database
                Prediction.objects.create(
                    input_data=json.dumps(input_data),
                    result=result
                )
                request.session['prediction_result'] = result
                return redirect('prediction_result')
    else:
        form = PredictionForm()
    
    # Filter history from database
    history = Prediction.objects.all().order_by('-created_at')
    
    # Apply filters
    if filter_result == 'fraud':
        history = history.filter(result='Fraud')
    elif filter_result == 'not_fraud':
        history = history.filter(result='Not Fraud')
    
    # Apply search if provided
    if search_query:
        history = history.filter(
            Q(result__icontains=search_query) |
            Q(input_data__icontains=search_query)
        )
    
    # Apply date filters
    if date_from:
        history = history.filter(created_at__gte=date_from)
    if date_to:
        history = history.filter(created_at__lte=date_to)
    
    # Get statistics
    total_predictions = Prediction.objects.count()
    fraud_count = Prediction.objects.filter(result='Fraud').count()
    not_fraud_count = Prediction.objects.filter(result='Not Fraud').count()
    
    return render(request, 'landing/dashboard.html', {
        'form': form,
        'result': result,
        'history': history,
        'errors': errors,
        'filter_result': filter_result,
        'search_query': search_query,
        'date_from': date_from,
        'date_to': date_to,
        'total_predictions': total_predictions,
        'fraud_count': fraud_count,
        'not_fraud_count': not_fraud_count,
        'filtered_count': history.count(),
    })

def landing_page(request):
    return render(request, 'landing/landing.html')

@login_required
def get_started_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data
            pred, errors = predict_fraud(input_data)
            if errors:
                return render(request, 'landing/get_started.html', {'form': form, 'errors': errors})
            result = 'Fraud' if pred == 1 else 'Not Fraud'
            # Optionally save to DB here
            request.session['prediction_result'] = result
            return redirect('prediction_result')
    else:
        form = PredictionForm()
    return render(request, 'landing/get_started.html', {'form': form})

@login_required
def prediction_result_view(request):
    result = request.session.get('prediction_result')
    if not result:
        return redirect('get_started')
    return render(request, 'landing/prediction_result.html', {'result': result})

# Authentication Views
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'landing/login.html')

def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Account created successfully! Welcome, {user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()
    
    return render(request, 'landing/signup.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login') 