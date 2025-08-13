from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from landing.models import Prediction, UserProfile
from django.utils import timezone
from datetime import timedelta
import json
import random

class Command(BaseCommand):
    help = 'Populate database with sample prediction data for testing dashboard'

    def handle(self, *args, **options):
        self.stdout.write('Creating sample prediction data...')
        
        # Get or create a test user
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        )
        
        if created:
            self.stdout.write(f'Created test user: {user.username}')
        
        # Sample input data for fraud detection
        sample_inputs = [
            {
                'Age': 25,
                'Gender': 'Male',
                'VehicleCategory': 'Sports Car',
                'PastNumberOfClaims': 3,
                'AnnualMileage': 15000,
                'MaritalStatus': 'Single',
                'Occupation': 'Student',
                'EducationLevel': 'High School',
                'Income': 25000,
                'CreditScore': 650,
                'DrivingRecord': 'Poor',
                'VehicleAge': 2,
                'PolicyType': 'Comprehensive',
                'CoverageAmount': 50000,
                'Deductible': 1000,
                'PreviousInsurance': 'No',
                'ClaimsHistory': 'Multiple',
                'VehicleValue': 35000,
                'PolicyDuration': 6,
                'PaymentMethod': 'Credit Card'
            },
            {
                'Age': 45,
                'Gender': 'Female',
                'VehicleCategory': 'SUV',
                'PastNumberOfClaims': 0,
                'AnnualMileage': 8000,
                'MaritalStatus': 'Married',
                'Occupation': 'Professional',
                'EducationLevel': 'Bachelor',
                'Income': 75000,
                'CreditScore': 780,
                'DrivingRecord': 'Excellent',
                'VehicleAge': 5,
                'PolicyType': 'Liability',
                'CoverageAmount': 100000,
                'Deductible': 500,
                'PreviousInsurance': 'Yes',
                'ClaimsHistory': 'None',
                'VehicleValue': 25000,
                'PolicyDuration': 12,
                'PaymentMethod': 'Bank Transfer'
            },
            {
                'Age': 32,
                'Gender': 'Male',
                'VehicleCategory': 'Sedan',
                'PastNumberOfClaims': 1,
                'AnnualMileage': 12000,
                'MaritalStatus': 'Divorced',
                'Occupation': 'Sales',
                'EducationLevel': 'Associate',
                'Income': 45000,
                'CreditScore': 720,
                'DrivingRecord': 'Good',
                'VehicleAge': 3,
                'PolicyType': 'Comprehensive',
                'CoverageAmount': 75000,
                'Deductible': 750,
                'PreviousInsurance': 'Yes',
                'ClaimsHistory': 'One',
                'VehicleValue': 30000,
                'PolicyDuration': 6,
                'PaymentMethod': 'Credit Card'
            }
        ]
        
        # Create sample predictions
        results = ['Fraud', 'Not Fraud']
        created_count = 0
        
        for i in range(20):  # Create 20 sample predictions
            # Randomly select input data and result
            input_data = random.choice(sample_inputs)
            result = random.choice(results)
            
            # Add some variation to the data
            input_data['Age'] = random.randint(18, 70)
            input_data['AnnualMileage'] = random.randint(5000, 25000)
            input_data['Income'] = random.randint(20000, 150000)
            input_data['CreditScore'] = random.randint(500, 850)
            
            # Create prediction with random timestamp in the last 30 days
            days_ago = random.randint(0, 30)
            created_at = timezone.now() - timedelta(days=days_ago)
            
            prediction = Prediction.objects.create(
                user=user,
                input_data=json.dumps(input_data),
                result=result,
                created_at=created_at,
                confidence_score=random.uniform(0.7, 0.95),
                processing_time=random.uniform(0.5, 3.0)
            )
            created_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {created_count} sample predictions for user {user.username}'
            )
        )
        
        # Display some statistics
        total_predictions = Prediction.objects.count()
        fraud_count = Prediction.objects.filter(result='Fraud').count()
        clean_count = Prediction.objects.filter(result='Not Fraud').count()
        
        self.stdout.write(f'Total predictions in database: {total_predictions}')
        self.stdout.write(f'Fraud cases: {fraud_count}')
        self.stdout.write(f'Clean cases: {clean_count}')
        
        self.stdout.write(
            self.style.SUCCESS('Sample data population completed successfully!')
        )
