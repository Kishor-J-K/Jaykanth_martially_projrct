import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Load the processed data
print("Loading data...")
df = pd.read_csv('martially_dataset_processed.csv')
print(f"Dataset shape: {df.shape}")

# Separate features and target
target = 'recommended_style'
X = df.drop(columns=[target])
y = df[target]

print(f"\nTarget classes: {y.unique()}")
print(f"Target distribution:\n{y.value_counts()}")

# Identify categorical and numerical columns
categorical_cols = ['experience', 'time_per_week', 'budget_range', 'format', 'location']
binary_cols = [col for col in X.columns if col not in categorical_cols]

print(f"\nCategorical features: {categorical_cols}")
print(f"Binary features (styles & goals): {len(binary_cols)} columns")

# Encode categorical features
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"\n{col} encoding:")
    for i, label in enumerate(le.classes_):
        print(f"  {i}: {label}")

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"\nTarget encoding:")
for i, label in enumerate(target_encoder.classes_):
    print(f"  {i}: {label}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Random Forest model
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate on training set
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

# Evaluate on test set
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report
print("\n" + "="*50)
print("Classification Report:")
print("="*50)
print(classification_report(y_test, y_test_pred, 
                          target_names=target_encoder.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("Top 15 Most Important Features:")
print("="*50)
print(feature_importance.head(15).to_string(index=False))

# Save the model and encoders
print("\n" + "="*50)
print("Saving model and encoders...")
print("="*50)

joblib.dump(rf_model, 'style_recommender_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

# Save feature names for reference
feature_info = {
    'categorical_cols': categorical_cols,
    'binary_cols': binary_cols,
    'all_features': list(X_encoded.columns)
}

with open('feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("Model saved as: style_recommender_model.pkl")
print("Encoders saved as: label_encoders.pkl, target_encoder.pkl")
print("Feature info saved as: feature_info.json")

print("\n" + "="*50)
print("Training completed successfully!")
print("="*50)

