import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_counts = df['Class/ASD'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(class_counts, labels=['No ASD (0)', 'ASD (1)'], autopct='%1.1f%%', colors=colors, explode=(0, 0.1))
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')

sns.countplot(data=df, x='Class/ASD', palette=colors, ax=axes[1])
axes[1].set_xlabel('Class/ASD', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['No ASD (0)', 'ASD (1)'])

plt.tight_layout()
plt.show()

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

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
confusion_matrices = {}

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
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No ASD', 'ASD'], yticklabels=['No ASD', 'ASD'])
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.suptitle('Confusion Matrices for All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

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

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.2

bars1 = ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy', color='#3498db')
bars2 = ax.bar(x - 0.5*width, results_df['Precision'], width, label='Precision', color='#2ecc71')
bars3 = ax.bar(x + 0.5*width, results_df['Recall'], width, label='Recall', color='#e74c3c')
bars4 = ax.bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score', color='#9b59b6')

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15)
ax.legend(loc='lower right')
ax.set_ylim(0, 1)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("📊 FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

print("\nTop 10 Most Important Features:")
print(feature_importance.tail(10).sort_values('Importance', ascending=False).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_importance)))
ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')

for i, v in enumerate(feature_importance['Importance']):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✅ AUTISM PREDICTION ANALYSIS COMPLETE!")
print("=" * 60)
