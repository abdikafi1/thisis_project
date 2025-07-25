from django.shortcuts import render, redirect
from .forms import PredictionForm
import pandas as pd
import joblib
import os
from .models import Prediction
import json

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

def prediction_view(request):
    return redirect('dashboard')

def history_view(request):
    history = Prediction.objects.all().order_by('-created_at')
    return render(request, 'landing/history.html', {'history': history})

def dashboard_view(request):
    result = None
    errors = None
    filter_result = request.GET.get('filter_result', 'all')
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
    if filter_result == 'fraud':
        history = Prediction.objects.filter(result='Fraud').order_by('-created_at')
    elif filter_result == 'not_fraud':
        history = Prediction.objects.filter(result='Not Fraud').order_by('-created_at')
    else:
        history = Prediction.objects.all().order_by('-created_at')
    return render(request, 'landing/dashboard.html', {
        'form': form,
        'result': result,
        'history': history,
        'errors': errors,
        'filter_result': filter_result
    })

def landing_page(request):
    return render(request, 'landing/landing.html')

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


def prediction_result_view(request):
    result = request.session.get('prediction_result')
    if not result:
        return redirect('get_started')
    return render(request, 'landing/prediction_result.html', {'result': result}) 