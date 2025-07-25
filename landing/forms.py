import pandas as pd
from django import forms
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'selected_fraud_and_4k_nonfraud.csv')
df = pd.read_csv(CSV_PATH)

def get_choices(col):
    return [(v, v) for v in sorted(df[col].dropna().unique())]

class PredictionForm(forms.Form):
    Days_Policy_Accident = forms.ChoiceField(
        choices=get_choices('Days_Policy_Accident'),
        help_text="Select the range of days since the policy accident.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    Days_Policy_Claim = forms.ChoiceField(
        choices=get_choices('Days_Policy_Claim'),
        help_text="Select the range of days since the policy claim.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    PastNumberOfClaims = forms.ChoiceField(
        choices=get_choices('PastNumberOfClaims'),
        help_text="Select the number of past claims.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    VehiclePrice = forms.ChoiceField(
        choices=get_choices('VehiclePrice'),
        help_text="Select the price range of the vehicle.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    PoliceReportFiled = forms.ChoiceField(
        choices=get_choices('PoliceReportFiled'),
        help_text="Was a police report filed?",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    WitnessPresent = forms.ChoiceField(
        choices=get_choices('WitnessPresent'),
        help_text="Was a witness present?",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    NumberOfSuppliments = forms.ChoiceField(
        choices=get_choices('NumberOfSuppliments'),
        help_text="Select the number of supplements.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    AddressChange_Claim = forms.ChoiceField(
        choices=get_choices('AddressChange_Claim'),
        help_text="Select the address change period for the claim.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    Deductible = forms.IntegerField(
        help_text="Enter the deductible amount (e.g., 400, 500, 700).",
        widget=forms.NumberInput(attrs={'placeholder': '400', 'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    DriverRating = forms.IntegerField(
        help_text="Enter the driver rating (1-4).",
        widget=forms.NumberInput(attrs={'placeholder': '1', 'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    Age = forms.IntegerField(
        help_text="Enter the age of the driver.",
        widget=forms.NumberInput(attrs={'placeholder': '35', 'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    AgeOfVehicle = forms.ChoiceField(
        choices=get_choices('AgeOfVehicle'),
        help_text="Select the age of the vehicle.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    Fault = forms.ChoiceField(
        choices=get_choices('Fault'),
        help_text="Who was at fault?",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    AccidentArea = forms.ChoiceField(
        choices=get_choices('AccidentArea'),
        help_text="Select the accident area.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    BasePolicy = forms.ChoiceField(
        choices=get_choices('BasePolicy'),
        help_text="Select the base policy type.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    VehicleCategory = forms.ChoiceField(
        choices=get_choices('VehicleCategory'),
        help_text="Select the vehicle category.",
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    ) 