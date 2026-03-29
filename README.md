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

The ROC curve shows that the model can distinguish between classes quite well.
The AUC score is around 0.81, which indicates good performance.

![ROC Curve](ROC.png)

## Model Comparison

| Model                       | Hyperparameters                             |   Train ROC-AUC |   Test ROC-AUC |   Train F1 |   Test F1 | Comment                                       |
|:----------------------------|:--------------------------------------------|----------------:|---------------:|-----------:|----------:|:----------------------------------------------|
| Logistic Regression (tuned) | C=1.0, class_weight=balanced, threshold=0.6 |           0.795 |          0.8   |      0.27  |     0.27  | Good baseline, stable                         |
| kNN (k=80)                  | n_neighbors=80                              |           0.92  |          0.78  |      0.48  |     0.32  | Overfitting reduced, but weaker than Log.Reg. |
| Decision Tree (depth=7)     | max_depth=7                                 |           0.8   |          0.796 |      0.44  |     0.41  | Good balance, interpretable                   |
| LightGBM (RandomSearch)     | RandomizedSearchCV tuned                    |           0.85  |          0.814 |      0.43  |     0.38  | Strong performance                            |
| LightGBM (Hyperopt)         | Hyperopt tuned                              |           0.846 |          0.816 |      0.434 |     0.388 | Best overall model                            |

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
