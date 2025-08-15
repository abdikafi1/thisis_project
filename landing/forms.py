import pandas as pd
from django import forms
import os
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, Prediction, UserActivity, SystemSettings

# Try to load CSV data, fallback to empty choices if not available
try:
    CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'selected_fraud_and_4k_nonfraud.csv')
    df = pd.read_csv(CSV_PATH)
    
    def get_choices(col):
        choices = [(v, v) for v in sorted(df[col].dropna().unique())]
        return choices
except:
    # Fallback if CSV or pandas is not available
    def get_choices(col):
        # Return basic choices for common fields
        basic_choices = {
            'Days_Policy_Accident': [
                ('1 to 7', '1 to 7'), ('8 to 15', '8 to 15'), ('16 to 30', '16 to 30'),
                ('31 to 60', '31 to 60'), ('61 to 90', '61 to 90'), ('91 to 180', '91 to 180')
            ],
            'Days_Policy_Claim': [
                ('1 to 7', '1 to 7'), ('8 to 15', '8 to 15'), ('16 to 30', '16 to 30'),
                ('31 to 60', '31 to 60'), ('61 to 90', '61 to 90'), ('91 to 180', '91 to 180')
            ],
            'PastNumberOfClaims': [
                ('none', 'none'), ('1', '1'), ('2 to 4', '2 to 4'), ('more than 4', 'more than 4')
            ],
            'VehiclePrice': [
                ('less than 20000', 'less than 20000'), ('20000 to 29000', '20000 to 29000'),
                ('30000 to 39000', '30000 to 39000'), ('40000 to 59000', '40000 to 59000'),
                ('60000 to 69000', '60000 to 69000'), ('more than 69000', 'more than 69000')
            ],
            'PoliceReportFiled': [('Yes', 'Yes'), ('No', 'No')],
            'WitnessPresent': [('Yes', 'Yes'), ('No', 'No')],
            'NumberOfSuppliments': [
                ('none', 'none'), ('1 to 2', '1 to 2'), ('3 to 5', '3 to 5'), ('more than 5', 'more than 5')
            ],
            'AddressChange_Claim': [
                ('no change', 'no change'), ('under 6 months', 'under 6 months'),
                ('1 year', '1 year'), ('2 to 3 years', '2 to 3 years'), ('4 to 8 years', '4 to 8 years')
            ],
            'AgeOfVehicle': [
                ('less than 1 year', 'less than 1 year'), ('1 to 2 years', '1 to 2 years'),
                ('3 to 5 years', '3 to 5 years'), ('6 to 10 years', '6 to 10 years'),
                ('more than 10 years', 'more than 10 years')
            ],
            'Fault': [('Policy Holder', 'Policy Holder'), ('Third Party', 'Third Party')],
            'AccidentArea': [('Urban', 'Urban'), ('Rural', 'Rural')],
            'BasePolicy': [('Liability', 'Liability'), ('Collision', 'Collision'), ('All Perils', 'All Perils')],
            'VehicleCategory': [
                ('Sport', 'Sport'), ('Utility', 'Utility'), ('Family', 'Family'),
                ('Luxury', 'Luxury'), ('Economy', 'Economy')
            ]
        }
        return basic_choices.get(col, [('', 'Select option')])

class PredictionForm(forms.Form):
    # Form title for Fraud Prediction Detection
    form_title = "Fraud Prediction Detection"
    
    Days_Policy_Accident = forms.ChoiceField(
        label="Days Between Policy Start and Accident",
        choices=[('', 'Select timeframe between policy start and accident occurrence')] + get_choices('Days_Policy_Accident'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    Days_Policy_Claim = forms.ChoiceField(
        label="Days Policy Claim",
        choices=[('', 'Select the range of days since the policy claim.')] + get_choices('Days_Policy_Claim'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    PastNumberOfClaims = forms.ChoiceField(
        label="Previous Insurance Claims History",
        choices=[('', 'Select driver\'s previous claims record (affects risk assessment)')] + get_choices('PastNumberOfClaims'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    VehiclePrice = forms.ChoiceField(
        label="Vehicle Market Value",
        choices=[('', 'Select vehicle\'s current market value range (higher value = higher coverage)')] + get_choices('VehiclePrice'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    PoliceReportFiled = forms.ChoiceField(
        label="Police Report Status",
        choices=[('', 'Was an official police report filed for this accident?')] + get_choices('PoliceReportFiled'),
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
        choices=[
            ('', 'Select driver rating based on driving history'),
            ('1', '1 - Poor (Multiple violations, high risk)'),
            ('2', '2 - Average (Some violations, moderate risk)'),
            ('3', '3 - Good (Minor violations, mostly clean record)'),
            ('4', '4 - Excellent (No violations, clean record')
        ],
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
        label="Fault Assignment",
        choices=[('', 'Select who was determined to be at fault for the accident')] + get_choices('Fault'),
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none transition'})
    )
    
    AccidentArea = forms.ChoiceField(
        label="Accident Location Type",
        choices=[('', 'Select the type of area where the accident occurred')] + get_choices('AccidentArea'),
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

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    phone_number = forms.CharField(max_length=15, required=False)
    company = forms.CharField(max_length=100, required=False)
    position = forms.CharField(max_length=100, required=False)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            # Update profile with additional fields
            profile = user.profile
            profile.phone_number = self.cleaned_data.get('phone_number', '')
            profile.company = self.cleaned_data.get('company', '')
            profile.position = self.cleaned_data.get('position', '')
            profile.save()
        
        return user

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['phone_number', 'company', 'position']
        widgets = {
            'phone_number': forms.TextInput(attrs={'class': 'form-control'}),
            'company': forms.TextInput(attrs={'class': 'form-control'}),
            'position': forms.TextInput(attrs={'class': 'form-control'}),
        }

class AdminUserManagementForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['user_level', 'is_verified']
        widgets = {
            'user_level': forms.Select(attrs={'class': 'form-control'}),
            'is_verified': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class SystemSettingsForm(forms.ModelForm):
    class Meta:
        model = SystemSettings
        fields = ['key', 'value', 'description']
        widgets = {
            'key': forms.TextInput(attrs={'class': 'form-control'}),
            'value': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
        }

class UserSearchForm(forms.Form):
    search = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search users by username, email, or company...'
        })
    )
    user_level = forms.ChoiceField(
        choices=[('', 'All Levels')] + UserProfile.USER_LEVELS,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    is_verified = forms.ChoiceField(
        choices=[('', 'All'), ('True', 'Verified'), ('False', 'Not Verified')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    is_active = forms.ChoiceField(
        choices=[('', 'All'), ('True', 'Active'), ('False', 'Inactive')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class AdminDashboardForm(forms.Form):
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    activity_type = forms.ChoiceField(
        choices=[('', 'All Activities')] + UserActivity.ACTIVITY_TYPES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    ) 

class UserAccountForm(forms.ModelForm):
    """Form for editing user account information"""
    first_name = forms.CharField(
        max_length=30, 
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your first name'
        })
    )
    last_name = forms.CharField(
        max_length=30, 
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your last name'
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your email address'
        })
    )
    username = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your username'
        })
    )
    
    class Meta:
        model = UserProfile
        fields = ['phone_number', 'company', 'position']
        widgets = {
            'phone_number': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
                'placeholder': 'Enter your phone number'
            }),
            'company': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
                'placeholder': 'Enter your company name'
            }),
            'position': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
                'placeholder': 'Enter your job position'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields['first_name'].initial = user.first_name
            self.fields['last_name'].initial = user.last_name
            self.fields['email'].initial = user.email
            self.fields['username'].initial = user.username 

class UnifiedProfileForm(forms.ModelForm):
    """Unified form for editing both user profile and admin profile information"""
    # User model fields
    first_name = forms.CharField(
        max_length=30, 
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your first name'
        })
    )
    last_name = forms.CharField(
        max_length=30, 
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your last name'
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your email address'
        })
    )
    
    # UserProfile model fields
    phone_number = forms.CharField(
        max_length=15, 
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your phone number'
        })
    )
    company = forms.CharField(
        max_length=100, 
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your company name'
        })
    )
    position = forms.CharField(
        max_length=100, 
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg',
            'placeholder': 'Enter your position'
        })
    )
    
    # Admin-only fields
    user_level = forms.ChoiceField(
        choices=UserProfile.USER_LEVELS,
        required=False,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white shadow-sm hover:shadow-lg'
        })
    )
    is_verified = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2'
        })
    )
    
    class Meta:
        model = UserProfile
        fields = ['phone_number', 'company', 'position', 'user_level', 'is_verified']
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            # Set initial values from user model
            self.fields['first_name'].initial = user.first_name
            self.fields['last_name'].initial = user.last_name
            self.fields['email'].initial = user.email
            
            # Set initial values from user profile
            if hasattr(user, 'profile'):
                profile = user.profile
                self.fields['phone_number'].initial = profile.phone_number
                self.fields['company'].initial = profile.company
                self.fields['position'].initial = profile.position
                self.fields['user_level'].initial = profile.user_level
                self.fields['is_verified'].initial = profile.is_verified
    
    def save(self, commit=True):
        profile = super().save(commit=False)
        
        if commit:
            # Save user model changes
            user = profile.user
            user.first_name = self.cleaned_data['first_name']
            user.last_name = self.cleaned_data['last_name']
            user.email = self.cleaned_data['email']
            user.save()
            
            # Save profile changes
            profile.save()
        
        return profile 