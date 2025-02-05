from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pandas as pd

# read in data
#data = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final.xlsx')
#data = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final_cleaned.xlsx')
#data = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_feat_selec_spear.xlsx')
data = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_feat_selec_rfe.xlsx')


# set features and target variable
X = data.drop(columns=['id_patient', 'dementia'])
y = data['dementia']  

# Set up the XGBoost model with default settings
model = XGBClassifier()

# # parameters to be grid searched
# param_grid = {
#     'n_estimators': [70, 75, 80],
#     'learning_rate': [0.07, 0.08, 0.09],
#     'max_depth': [4, 5, 6],
#     'subsample': [0.50, 0.55, 0.60],
#     'colsample_bytree': [0.45, 0.50, 0.55]
# }

# # parameters to be grid searched - blank rows removed
# param_grid = {
#     'n_estimators': [75, 80, 85],
#     'learning_rate': [0.07, 0.08, 0.09],
#     'max_depth': [5, 6, 7],
#     'subsample': [0.70, 0.75, 0.80],
#     'colsample_bytree': [0.65, 0.70, 0.75]
# }

# # parameters to be grid searched spearman
# param_grid = {
#     'n_estimators': [100, 105, 110],
#     'learning_rate': [0.08, 0.09, 0.10],
#     'max_depth': [6, 7, 8],
#     'subsample': [0.60, 0.65, 0.70],
#     'colsample_bytree': [0.50, 0.55, 0.60]
# }

# parameters to be grid searched rfe
param_grid = {
    'n_estimators': [90, 95, 100],
    'learning_rate': [0.05, 0.06, 0.07],
    'max_depth': [4, 5, 6],
    'subsample': [0.70, 0.75, 0.80],
    'colsample_bytree': [0.60, 0.65, 0.70]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform Grid Search
grid_search.fit(X, y)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)




