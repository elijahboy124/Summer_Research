import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE

#data = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final.xlsx')
data = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final_cleaned.xlsx')


##
# Spearman Rank Correlation feature selection
##

# Extract features and target
X = data.drop(columns=["dementia", "id_patient"])
y = data["dementia"]

# Compute Spearman correlations
correlation_results = {}
for column in X.columns:
    correlation, _ = spearmanr(X[column], y)
    correlation_results[column] = correlation

#Display Spearman correlations
print("Spearman Correlations:")
for feature, corr in correlation_results.items():
    print(f"{feature}: {corr:.2f}")

# Select features with correlation above a threshold
threshold = 0.15 
selected_features = [feature for feature, corr in correlation_results.items() if abs(corr) >= threshold]


#selected_features.extend(["dementia", "id_patient", "TBV", "TIV", "TBV/TIV", "L_HC", "R_HC", "Combined_HC"])
selected_features.extend(["dementia", "id_patient"])

spearman_data = data[selected_features]

spearman_data.to_excel(r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_feat_selec_spear.xlsx", index=False)



##
# Recursive Feature Elimination
##

X = data.drop(columns=["dementia", "id_patient"])
y = data["dementia"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=105,       
    max_depth=5,            
    learning_rate=0.07,
    colsample_bytree=0.6,
    subsample=0.75,
    booster='gbtree',    
    eval_metric='logloss'
)


# Applying RFE to select 20 features 
rfe = RFE(estimator=model, n_features_to_select=20, step=1)

# Fit RFE
rfe.fit(X_train, y_train)

# Get the selected features from RFE
selected_features_rfe = X.columns[rfe.support_]
selected_features_rfe_list = selected_features_rfe.tolist()
selected_features_rfe_list.extend(["dementia", "id_patient"])

# create final dataframe
final_data = data[selected_features_rfe_list]
final_data.to_excel(r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_feat_selec_rfe.xlsx", index=False)
