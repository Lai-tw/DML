
#強化 XGBoost + Feature Selection + F1 最佳化 
# + 隨機過採樣

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import pandas as pd
import numpy as np

# File paths
train_file = '38_Training_Data_Set_V2/training.csv'
test_file = '38_Public_Test_Set_and_Submmision_Template_V2/public_x.csv'
output_file = 'predictions_XGBoost_optimized.csv'

# Step 1: Load training data
print("Loading training data...")
train_df = pd.read_csv(train_file)
label_column = train_df.columns[-1]
X = train_df.drop(columns=['ID', label_column])
y = train_df[label_column]

# Clean data
print("Cleaning data...")
X = X.fillna(X.mean())
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.max().max())

# Step 2: Feature selection with XGBoost
print("Selecting top 1000 features...")
selector_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, eval_metric='auc')
selector_model.fit(X, y)
selector = SelectFromModel(selector_model, prefit=True, max_features=1000, threshold=-np.inf)
X = selector.transform(X)

# Step 3: Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Step 5: Handle class imbalance with RandomOverSampler
print("Applying RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

# Optional: Calculate scale_pos_weight if needed
pos = sum(y_train_resampled == 1)
neg = sum(y_train_resampled == 0)
scale_weight = neg / pos
print(f"scale_pos_weight: {scale_weight:.2f}")

# Step 6: Train XGBoost model
print("Training model...")
eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    colsample_bytree=0.7,
    subsample=0.7,
    random_state=42,
    scale_pos_weight=scale_weight
)

model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=eval_set,
    early_stopping_rounds=30,
    verbose=True
)

# Step 7: Threshold tuning for best F1 score
print("Tuning threshold for best F1 score...")
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
best_thresh, best_f1 = 0.5, 0

for thresh in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_val_proba > thresh).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"Best threshold: {best_thresh:.2f}, Best F1: {best_f1:.4f}")

# Step 8: Final evaluation on validation set
print("Evaluating on validation set with best threshold...")
y_pred_val = (y_val_proba > best_thresh).astype(int)
print("Classification Report:")
print(classification_report(y_val, y_pred_val))
auc_score = roc_auc_score(y_val, y_val_proba)
print(f"AUC-ROC Score: {auc_score:.4f}")

# Step 9: Load and preprocess test data
print("Loading test data...")
test_df = pd.read_csv(test_file)
X_test = test_df.drop(columns=['ID'])
X_test = X_test.fillna(X_test.mean())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.max().max())
X_test = selector.transform(X_test)
X_test_scaled = scaler.transform(X_test)

# Step 10: Predict with best threshold
print("Predicting on test set...")
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred_test = (y_test_proba > best_thresh).astype(int)

# Step 11: Save predictions
output_df = pd.DataFrame({
    'ID': test_df['ID'],
    label_column: y_pred_test
})
output_df.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'.")
