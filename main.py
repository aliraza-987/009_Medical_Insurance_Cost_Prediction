import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/medical_insurance.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n" + "="*60)
print("STEP 1: CLEAN DATA")
print("="*60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\n" + "="*60)
print("STEP 2: ENCODE CATEGORICAL DATA")
print("="*60)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(categorical_cols)}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\n" + "="*60)
print("STEP 3: PREPARE DATA")
print("="*60)

X = df.drop('charges', axis=1).values.astype(float)
y = df['charges'].values.astype(float)

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Insurance Cost - Min: ${y.min():.2f}, Max: ${y.max():.2f}, Mean: ${y.mean():.2f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

print("\n" + "="*60)
print("STEP 4: TRAIN MODELS")
print("="*60)

print("\n1. Training Linear Regression...")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"   Accuracy (R²): {r2_lr:.4f}")
print(f"   RMSE: ${rmse_lr:.2f}")
print(f"   MAE: ${mae_lr:.2f}")

print("\n2. Training Random Forest...")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"   Accuracy (R²): {r2_rf:.4f}")
print(f"   RMSE: ${rmse_rf:.2f}")
print(f"   MAE: ${mae_rf:.2f}")

print("\n3. Training Support Vector Regression...")
model_svr = SVR(kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"   Accuracy (R²): {r2_svr:.4f}")
print(f"   RMSE: ${rmse_svr:.2f}")
print(f"   MAE: ${mae_svr:.2f}")

print("\n" + "="*60)
print("STEP 5: SELECT BEST MODEL")
print("="*60)

models_dict = {
    'Linear Regression': (r2_lr, rmse_lr, mae_lr, y_pred_lr),
    'Random Forest': (r2_rf, rmse_rf, mae_rf, y_pred_rf),
    'Support Vector Regression': (r2_svr, rmse_svr, mae_svr, y_pred_svr)
}

best_model_name = max(models_dict, key=lambda x: models_dict[x][0])
best_r2 = models_dict[best_model_name][0]
best_rmse = models_dict[best_model_name][1]
best_mae = models_dict[best_model_name][2]
y_pred_best = models_dict[best_model_name][3]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"✓ R² Score: {best_r2:.4f}")
print(f"✓ RMSE: ${best_rmse:.2f}")
print(f"✓ MAE: ${best_mae:.2f}")

print("\n" + "="*60)
print("STEP 6: MODEL COMPARISON TABLE")
print("="*60)

print("\n| Model | R² Score | RMSE | MAE |")
print("|-------|----------|------|-----|")
print(f"| Linear Regression | {r2_lr:.4f} | ${rmse_lr:.2f} | ${mae_lr:.2f} |")
print(f"| Random Forest | {r2_rf:.4f} | ${rmse_rf:.2f} | ${mae_rf:.2f} |")
print(f"| Support Vector Regression | {r2_svr:.4f} | ${rmse_svr:.2f} | ${mae_svr:.2f} |")

print("\n" + "="*60)
print("STEP 7: SAMPLE PREDICTIONS")
print("="*60)

print("\nFirst 10 Test Predictions:")
print(f"\n{'#':<4} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
print("-" * 40)
for i in range(min(10, len(y_test))):
    actual = y_test[i]
    predicted = y_pred_best[i]
    error = abs(actual - predicted)
    print(f"{i+1:<4} ${actual:<11.2f} ${predicted:<11.2f} ${error:<9.2f}")

print("\n" + "="*60)
print("STEP 8: CREATE VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.6, color='green', s=30)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Insurance Cost ($)', fontsize=10)
axes[0, 0].set_ylabel('Predicted Insurance Cost ($)', fontsize=10)
axes[0, 0].set_title(f'Linear Regression (R² = {r2_lr:.4f})', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.6, color='blue', s=30)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Insurance Cost ($)', fontsize=10)
axes[0, 1].set_ylabel('Predicted Insurance Cost ($)', fontsize=10)
axes[0, 1].set_title(f'Random Forest (R² = {r2_rf:.4f})', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_test, y_pred_svr, alpha=0.6, color='orange', s=30)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Insurance Cost ($)', fontsize=10)
axes[1, 0].set_ylabel('Predicted Insurance Cost ($)', fontsize=10)
axes[1, 0].set_title(f'Support Vector Regression (R² = {r2_svr:.4f})', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(y_test, y_pred_best, alpha=0.6, color='purple', s=30)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Insurance Cost ($)', fontsize=10)
axes[1, 1].set_ylabel('Predicted Insurance Cost ($)', fontsize=10)
axes[1, 1].set_title(f'{best_model_name} - BEST (R² = {best_r2:.4f})', fontsize=11, fontweight='bold', color='green')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Graph saved as 'prediction_results.png'")
plt.show()

print("\n" + "="*60)
print("PROJECT COMPLETE!")
print("="*60)