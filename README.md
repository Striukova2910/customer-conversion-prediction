# customer-conversion-prediction
Machine learning model to predict customer conversion using classification models and boosting (LightGBM, Hyperopt, SHAP analysis)

## Overview
This project predicts whether a client will accept a bank offer (yes/no).
It is a binary classification problem based on client data, campaign information, and economic indicators.

## Models
- Logistic Regression  
- kNN  
- Decision Tree  
- LightGBM  

LightGBM was also tuned using:
- RandomizedSearchCV  
- Hyperopt  

## Results
Best model: LightGBM

- ROC-AUC ≈ 0.81  
- F1-score ≈ 0.38

  #%%
# results for GitHub
print(results.to_markdown())

## Key Insights
- Economic features (euribor3m, employment rate) are important  
- Campaign features (number of contacts) affect results  
- Previous client history has strong impact  


## SHAP Analysis
SHAP was used to explain model predictions.
It showed how each feature affects the probability of a positive outcome.


## Error Analysis
The model often makes mistakes for:
- clients with no previous contact history  
- cases where campaign impact is not clear  

## Tech Stack
Python, pandas, scikit-learn, LightGBM, Hyperopt, SHAP
