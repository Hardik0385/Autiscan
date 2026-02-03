import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("AUTISM PREDICTION USING MACHINE LEARNING")
print("=" * 60)

df = pd.read_csv('train.csv')

print("\n📊 DATASET INFORMATION")
print("-" * 40)
print(f"Total samples: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")

print("\n📋 COLUMN NAMES:")
print(df.columns.tolist())

print("\n🔍 FIRST 5 ROWS:")
print(df.head())

print("\n📈 DATA TYPES:")
print(df.dtypes)

print("\n📉 STATISTICAL SUMMARY:")
print(df.describe())

print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

print("\n🔍 Checking for '?' values (missing data):")
for col in df.columns:
    missing_count = (df[col] == '?').sum() if df[col].dtype == 'object' else 0
    if missing_count > 0:
        print(f"  {col}: {missing_count} missing values")

print("\n📊 CLASS DISTRIBUTION (Target Variable):")
print(df['Class/ASD'].value_counts())
print(f"\nClass balance: {df['Class/ASD'].value_counts(normalize=True).to_dict()}")

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

df_processed = df.copy()

for col in df_processed.columns:
    if df_processed[col].dtype == 'object':
        mode_value = df_processed[df_processed[col] != '?'][col].mode()
        if len(mode_value) > 0:
            df_processed[col] = df_processed[col].replace('?', mode_value[0])

print("✅ Replaced missing values ('?') with mode")

columns_to_drop = ['ID', 'age_desc', 'contry_of_res']
df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
print(f"✅ Dropped columns: {columns_to_drop}")

label_encoders = {}
categorical_columns = df_processed.select_dtypes(include=['object']).columns

print("\n🔄 Encoding categorical variables:")
for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  ✅ Encoded: {col}")

print(f"\n📊 Processed dataset shape: {df_processed.shape}")

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X = df_processed.drop('Class/ASD', axis=1)
y = df_processed['Class/ASD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Features used: {X.columns.tolist()}")

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🤖 {name}")
    print('='*50)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0][0]:3d}  FP={cm[0][1]:3d}]")
    print(f"   [FN={cm[1][0]:3d}  TP={cm[1][1]:3d}]]")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No ASD (0)', 'ASD (1)']))
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

print("\n" + "=" * 60)
print("📊 MODEL COMPARISON SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\n")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']*100:.2f}%")

print("\n" + "=" * 60)
print("📊 FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("✅ AUTISM PREDICTION ANALYSIS COMPLETE!")
print("=" * 60)
