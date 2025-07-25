import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
import joblib
import os

# Load data
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'selected_fraud_and_4k_nonfraud.csv')
df = pd.read_csv(csv_path)
y = df['FraudFound_P']
X = df.drop('FraudFound_P', axis=1)

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str).str.strip())
    encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = BalancedRandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save model, encoders, and categorical columns
save_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(save_dir, 'model.pkl'))
joblib.dump(encoders, os.path.join(save_dir, 'encoders.pkl'))
joblib.dump(categorical_cols, os.path.join(save_dir, 'categorical_cols.pkl'))
print('Model, encoders, and categorical columns saved to ml_model/') 