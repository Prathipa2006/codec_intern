# Credit Card Fraud Detection

## Goal
Detect fraudulent credit card transactions using classification and anomaly detection techniques.

## Dataset
- Source: Kaggle "Credit Card Fraud Detection" (mlg-ulb)
- 284,807 transactions, only 492 fraud (0.17%) — highly imbalanced
- Features: V1-V28 (PCA-anonymized), Time, Amount, Class (target)

## Approach
1. Scaled `Amount` and `Time` using StandardScaler
2. Split data 80/20, stratified by class
3. Balanced training data using SMOTE
4. Trained and compared 3 models:
   - Random Forest (classification)
   - XGBoost (classification)
   - Isolation Forest (unsupervised anomaly detection)

## Results (on fraud class)
| Model | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|
| Random Forest | 0.82 | 0.82 | 0.82 | 0.969 |
| XGBoost | 0.68 | 0.86 | 0.76 | 0.975 |
| Isolation Forest | 0.65 | 0.66 | 0.66 | - |

## Key Insight
Accuracy is misleading here (99.8%+ for any model) due to extreme class imbalance.
F1-score and AUC-ROC were used instead. Supervised models (Random Forest, XGBoost)
outperformed unsupervised anomaly detection (Isolation Forest), since they could
learn from labeled fraud examples. XGBoost was chosen as the final model for its
higher fraud recall and AUC-ROC.

## Files
- `main.ipynb` - full analysis and modeling
- `fraud_model.pkl` - saved XGBoost model