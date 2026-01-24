import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import shap

DATA_PATH = "../data/diabetes_dataset.csv"
MODEL_DIR = "../models"

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    
    df = df.drop_duplicates()
    df['smoking_history'] = df['smoking_history'].replace(['No Info', 'no info', 'None'], 'unknown')
    df['smoking_history'] = df['smoking_history'].fillna('unknown')
    
    return df

def encode_features(df):
    label_encoders = {}
    df_encoded = df.copy()
    
    categorical_cols = ['gender', 'smoking_history']
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str).str.lower())
        label_encoders[col] = le
    
    return df_encoded, label_encoders

def train_model():
    print("Loading data...")
    df = load_and_preprocess_data()
    
    print("Encoding features...")
    df_encoded, label_encoders = encode_features(df)
    
    feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 
                    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    X = df_encoded[feature_cols]
    y = df_encoded['diabetes'].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\nGenerating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{MODEL_DIR}/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open(f"{MODEL_DIR}/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    
    with open(f"{MODEL_DIR}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    dataset_stats = {
        'total_samples': len(df),
        'diabetic_samples': int(df['diabetes'].sum()),
        'non_diabetic_samples': int(len(df) - df['diabetes'].sum()),
        'feature_ranges': {
            'age': {'min': float(df['age'].min()), 'max': float(df['age'].max()), 'mean': float(df['age'].mean())},
            'bmi': {'min': float(df['bmi'].min()), 'max': float(df['bmi'].max()), 'mean': float(df['bmi'].mean())},
            'HbA1c_level': {'min': float(df['HbA1c_level'].min()), 'max': float(df['HbA1c_level'].max()), 'mean': float(df['HbA1c_level'].mean())},
            'blood_glucose_level': {'min': float(df['blood_glucose_level'].min()), 'max': float(df['blood_glucose_level'].max()), 'mean': float(df['blood_glucose_level'].mean())}
        },
        'smoking_categories': df['smoking_history'].unique().tolist(),
        'gender_categories': df['gender'].unique().tolist()
    }
    
    with open(f"{MODEL_DIR}/dataset_stats.pkl", "wb") as f:
        pickle.dump(dataset_stats, f)
    
    print("\nModel and artifacts saved successfully!")
    print(f"Files saved in: {MODEL_DIR}")
    
    return model, label_encoders, scaler, explainer

if __name__ == "__main__":
    train_model()
