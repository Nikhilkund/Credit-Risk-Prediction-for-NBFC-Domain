# Credit-Risk-Prediction-for-NBFC-Domain

This repository contains a machine learning model for predicting credit risk. The model is trained on a pre-processed dataset (not included in this repo), using Logistic Regression optimized with Optuna. The best-performing model, based on F1-score and other evaluation metrics, is saved for deployment.

## Model Details

- **Model Type:** Logistic Regression
- **Feature Engineering:**
  - Features are pre-processed and encoded.
  - Categorical features are one-hot encoded.
  - Numerical features are scaled.
- **Class Imbalance Handling:** SMOTETomek is used to oversample the minority class and undersample the majority class.
- **Hyperparameter Tuning:** Optuna is used for Bayesian optimization to find the best hyperparameters for Logistic Regression.
- **Evaluation Metrics:**
  - Precision
  - Recall
  - F1-score
  - AUC (Area Under the ROC Curve)
  - Gini Coefficient
  - KS Statistic (Kolmogorov-Smirnov)
  
## Results

The best Logistic Regression model achieved the following performance metrics (results may vary depending on your dataset):

AUC: ~0.98
Gini Coefficient: ~0.96
KS Statistic: ~80-85%, indicating good separation between good and bad customers.


## Further Improvements 

Explore other models such as Gradient Boosting or Support Vector Machines.
Apply advanced feature engineering.
Enhance hyperparameter tuning with more trials in Optuna.
Consider ensemble methods for improved performance.
Focus on model explainability using techniques like SHAP or LIME.


## Requirements
optuna
scikit-learn
xgboost
imblearn
pandas
matplotlib
joblib
numpy
scipy
