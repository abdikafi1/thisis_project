import pandas as pd
from django import forms
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'selected_fraud_and_4k_nonfraud.csv')
df = pd.read_csv(CSV_PATH)

def get_choices(col):
    choices = [(v, v) for v in sorted(df[col].dropna().unique())]
    return choices

class PredictionForm(forms.Form):
    # Form title for Fraud Prediction Detection
    form_title = "Fraud Prediction Detection"
    
    Days_Policy_Accident = forms.ChoiceField(
        label="Days Policy Accident",
        choices=[('', 'Select the range of days since the policy accident.')] + get_choices('Days_Policy_Accident'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    Days_Policy_Claim = forms.ChoiceField(
        label="Days Policy Claim",
        choices=[('', 'Select the range of days since the policy claim.')] + get_choices('Days_Policy_Claim'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    PastNumberOfClaims = forms.ChoiceField(
        label="Past Number Of Claims",
        choices=[('', 'Select the number of past claims.')] + get_choices('PastNumberOfClaims'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    VehiclePrice = forms.ChoiceField(
        label="Vehicle Price",
        choices=[('', 'Select the price range of the vehicle.')] + get_choices('VehiclePrice'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    PoliceReportFiled = forms.ChoiceField(
        label="Police Report Filed",
        choices=[('', 'Select whether a police report was filed.')] + get_choices('PoliceReportFiled'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    WitnessPresent = forms.ChoiceField(
        label="Witness Present",
        choices=[('', 'Select whether a witness was present.')] + get_choices('WitnessPresent'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    NumberOfSuppliments = forms.ChoiceField(
        label="Number Of Suppliments",
        choices=[('', 'Select the number of supplements.')] + get_choices('NumberOfSuppliments'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    AddressChange_Claim = forms.ChoiceField(
        label="Address Change Claim",
        choices=[('', 'Select the address change period for the claim.')] + get_choices('AddressChange_Claim'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    Deductible = forms.IntegerField(
        label="Deductible Amount",
        help_text="Enter the deductible amount (e.g., 400, 500, 700).",
        widget=forms.NumberInput(attrs={'placeholder': 'Enter deductible amount (400-1000)', 'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    DriverRating = forms.ChoiceField(
        label="Driver Rating",
        choices=[('', 'Select the driver rating from 1 to 4.')] + [(str(i), str(i)) for i in range(1, 5)],
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    Age = forms.IntegerField(
        label="Driver Age",
        help_text="Enter the age of the driver (18-71).",
        widget=forms.NumberInput(attrs={'placeholder': 'Enter age (18-71)', 'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    AgeOfVehicle = forms.ChoiceField(
        label="Age Of Vehicle",
        choices=[('', 'Select the age of the vehicle.')] + get_choices('AgeOfVehicle'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    Fault = forms.ChoiceField(
        label="Fault",
        choices=[('', 'Select who was at fault.')] + get_choices('Fault'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    AccidentArea = forms.ChoiceField(
        label="Accident Area",
        choices=[('', 'Select the accident area.')] + get_choices('AccidentArea'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    BasePolicy = forms.ChoiceField(
        label="Base Policy",
        choices=[('', 'Select the base policy type.')] + get_choices('BasePolicy'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    VehicleCategory = forms.ChoiceField(
        label="Vehicle Category",
        choices=[('', 'Select the vehicle category.')] + get_choices('VehicleCategory'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    def clean_Age(self):
        age = self.cleaned_data.get('Age')
        if age is None:
            raise forms.ValidationError("Age is required.")
        
        # Check if age is a double/decimal value
        if isinstance(age, float):
            if age < 18 or age > 71:
                raise forms.ValidationError("Age must be between 18 and 71 (no decimal values allowed).")
            if age != int(age):
                raise forms.ValidationError("Age must be a whole number (no decimal values).")
            return int(age)  # Convert to integer
        elif isinstance(age, int):
            if age < 18 or age > 71:
                raise forms.ValidationError("Age must be between 18 and 71.")
            return age
        else:
            raise forms.ValidationError("Age must be a valid number.")
    
    def clean_DriverRating(self):
        rating = self.cleaned_data.get('DriverRating')
        if not rating:
            raise forms.ValidationError("Please select a driver rating.")
        return rating
    
    def clean_Deductible(self):
        deductible = self.cleaned_data.get('Deductible')
        if deductible is None:
            raise forms.ValidationError("Deductible amount is required.")
        
        # Check if deductible is a double/decimal value
        if isinstance(deductible, float):
            if deductible < 0:
                raise forms.ValidationError("Deductible cannot be negative.")
            if deductible > 1000:
                raise forms.ValidationError("Deductible cannot exceed 1000.")
            if deductible != int(deductible):
                raise forms.ValidationError("Deductible must be a whole number (no decimal values).")
            return int(deductible)  # Convert to integer
        elif isinstance(deductible, int):
            if deductible < 0:
                raise forms.ValidationError("Deductible cannot be negative.")
            if deductible > 1000:
                raise forms.ValidationError("Deductible cannot exceed 1000.")
            return deductible
        else:
            raise forms.ValidationError("Deductible must be a valid number.")
    
    def clean_Days_Policy_Accident(self):
        days = self.cleaned_data.get('Days_Policy_Accident')
        if not days:
            raise forms.ValidationError("Please select the days since policy accident.")
        return days
    
    def clean_Days_Policy_Claim(self):
        days = self.cleaned_data.get('Days_Policy_Claim')
        if not days:
            raise forms.ValidationError("Please select the days since policy claim.")
        return days
    
    def clean_PastNumberOfClaims(self):
        claims = self.cleaned_data.get('PastNumberOfClaims')
        if not claims:
            raise forms.ValidationError("Please select the number of past claims.")
        return claims
    
    def clean_VehiclePrice(self):
        price = self.cleaned_data.get('VehiclePrice')
        if not price:
            raise forms.ValidationError("Please select the vehicle price range.")
        return price
    
    def clean_PoliceReportFiled(self):
        report = self.cleaned_data.get('PoliceReportFiled')
        if not report:
            raise forms.ValidationError("Please indicate if a police report was filed.")
        return report
    
    def clean_WitnessPresent(self):
        witness = self.cleaned_data.get('WitnessPresent')
        if not witness:
            raise forms.ValidationError("Please indicate if a witness was present.")
        return witness
    
    def clean_NumberOfSuppliments(self):
        supplements = self.cleaned_data.get('NumberOfSuppliments')
        if not supplements:
            raise forms.ValidationError("Please select the number of supplements.")
        return supplements
    
    def clean_AddressChange_Claim(self):
        address_change = self.cleaned_data.get('AddressChange_Claim')
        if not address_change:
            raise forms.ValidationError("Please select the address change period.")
        return address_change
    
    def clean_AgeOfVehicle(self):
        age = self.cleaned_data.get('AgeOfVehicle')
        if not age:
            raise forms.ValidationError("Please select the age of the vehicle.")
        return age
    
    def clean_Fault(self):
        fault = self.cleaned_data.get('Fault')
        if not fault:
            raise forms.ValidationError("Please select who was at fault.")
        return fault
    
    def clean_AccidentArea(self):
        area = self.cleaned_data.get('AccidentArea')
        if not area:
            raise forms.ValidationError("Please select the accident area.")
        return area
    
    def clean_BasePolicy(self):
        policy = self.cleaned_data.get('BasePolicy')
        if not policy:
            raise forms.ValidationError("Please select the base policy type.")
        return policy
    
    def clean_VehicleCategory(self):
        category = self.cleaned_data.get('VehicleCategory')
        if not category:
            raise forms.ValidationError("Please select the vehicle category.")
        return category
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Cross-field validation
        days_accident = cleaned_data.get('Days_Policy_Accident')
        days_claim = cleaned_data.get('Days_Policy_Claim')
        
        if days_accident and days_claim:
            # Validate that claim days should not be less than accident days
            try:
                accident_days = int(days_accident.split()[0]) if 'to' in days_accident else int(days_accident)
                claim_days = int(days_claim.split()[0]) if 'to' in days_claim else int(days_claim)
                
                if claim_days < accident_days:
                    raise forms.ValidationError("Claim days cannot be less than accident days.")
            except (ValueError, IndexError):
                pass  # Skip validation if parsing fails
        
        return cleaned_data 