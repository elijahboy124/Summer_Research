import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBClassifier
import shap
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt

# check without the dementia treatment feature

columns = ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1"]
indexs = ["Original", "Original no CT", "Original no ARC", "Original no CT or ARC", "Cleaned", "Cleaned no CT", "Cleaned no ARC", "Cleaned no CT or ARC",
          "Spearman", "Spearman no CT", "Spearman no ARC", "Spearman no CT or ARC", "RFE", "RFE no CT", "RFE no ARC", "RFE no CT or ARC"]
df_output = pd.DataFrame(columns=columns, index=indexs)

# read in data without feature selection
data1 = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final.xlsx')
data2 = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final_cleaned.xlsx')
data3 = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_feat_selec_spear.xlsx')
data4 = pd.read_excel(r'C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_feat_selec_rfe.xlsx')

datas = [data1, data2, data3, data4]

# hyperparameters without feature selection
model1 = XGBClassifier(n_estimators=75, max_depth=5, learning_rate=0.08, colsample_bytree=0.5, subsample=0.55, booster='gbtree', eval_metric='logloss')
# hyperparameters without feature selection blank rows removed
model2 = XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.08, colsample_bytree=0.7, subsample=0.75, booster='gbtree', eval_metric='logloss')
# hyperparameters with spearman feature selection
model3 = XGBClassifier(n_estimators=105, max_depth=7, learning_rate=0.09, colsample_bytree=0.55, subsample=0.65, booster='gbtree', eval_metric='logloss')
# hyperparameters with rfe feature selection
model4 = XGBClassifier(n_estimators=95, max_depth=5, learning_rate=0.06, colsample_bytree=0.65, subsample=0.75, booster='gbtree', eval_metric='logloss')
models = [model1, model2, model3, model4]

count = 0

for data, model in zip(datas, models):
    for i in range(4):
        if i == 0:
            # select all features
            X = data.drop(columns=['id_patient', 'dementia', 'Treatments for Dementia - Pharmacy'], errors='ignore')
        elif i == 1:
            # select features without CT data
            X = data.drop(columns=['id_patient', 'dementia', 'TBV', 'TIV', 'TBV/TIV', 'L_HC', 'R_HC', 'Combined_HC', 'HC/TIV', 'Treatments for Dementia - Pharmacy'], errors='ignore')
        elif i == 2:
            # select features without ARC data
            X = data.drop(columns=['id_patient', 'dementia', 'Dementia Stay Days - ARC', 'Hospital Stay Days - ARC', 'Psycgeri Stay Days - ARC', 'Resthome Stay Days - ARC', 'Treatments for Dementia - Pharmacy'], errors='ignore')
        elif i == 3:
            # select features without CT data or ARC data
            X = data.drop(columns=['id_patient', 'dementia', 'TBV', 'TIV', 'TBV/TIV', 'L_HC', 'R_HC', 'Combined_HC', 'HC/TIV', 'Dementia Stay Days - ARC', 'Hospital Stay Days - ARC', 'Psycgeri Stay Days - ARC', 'Resthome Stay Days - ARC', 'Treatments for Dementia - Pharmacy'], errors='ignore')
        
        # select target feature
        y = data['dementia'] 

        # Define the k-fold cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        ix_training, ix_test = [], []
        # Loop through each fold and append the training & test indices to the empty lists above
        for fold in kf.split(X):
            ix_training.append(fold[0])
            ix_test.append(fold[1])

        # Initialize lists to store metric values for each fold
        accuracies = []
        sensitivities = []
        specificities = []
        f1_scores = []
        aucs = []

        shap_values_per_fold = []
        
        # Perform k-fold cross-validation manually
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model on training data
            model.fit(X_train, y_train)

            # get shap plot data for current model and fold
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # store shap values into list of lists
            for shaps in shap_values:
                shap_values_per_fold.append(shaps)

            # Make predictions for the test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

            # Accuracy for this fold
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

            # Confusion matrix and metrics for this fold
            conf_matrix = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()

            # Sensitivity (Recall)
            sensitivity = tp / (tp + fn)
            sensitivities.append(sensitivity)

            # Specificity
            specificity = tn / (tn + fp)
            specificities.append(specificity)

            # F1-Score
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)

            # AUC for this fold
            auc = roc_auc_score(y_test, y_prob)
            aucs.append(auc)
        
        new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]

        plt.figure()
        shap.summary_plot(np.array(shap_values_per_fold), X.reindex(new_index), show=False, plot_size=(16,8))
        plt.tight_layout()
        plt.title(f"Shap Summary Plot for Model: {indexs[count]}")
        if data is data1:
            plot_path = f'C:\\Users\\ehay055\\OneDrive - The University of Auckland\\Desktop\\research\\model\\plots\\no_dementia_treatment\\{indexs[count]}.png'
        elif data is data2:
            plot_path = f'C:\\Users\\ehay055\\OneDrive - The University of Auckland\\Desktop\\research\\model\\plots\\no_dementia_treatment\\{indexs[count]}.png'
        elif data is data3:
            plot_path = f'C:\\Users\\ehay055\\OneDrive - The University of Auckland\\Desktop\\research\\model\\plots\\no_dementia_treatment\\{indexs[count]}.png'
        elif data is data4:
            plot_path = f'C:\\Users\\ehay055\\OneDrive - The University of Auckland\\Desktop\\research\\model\\plots\\no_dementia_treatment\\{indexs[count]}.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        #print(f"Processing model: {indexs[count]}, count: {count}")
        count += 1
        
        results = []
        # After cross-validation, calculate the mean and standard deviation for each metric
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        results.append(f'{mean_accuracy:.3f} ± {std_accuracy:.3f}')

        mean_sensitivity = np.mean(sensitivities)
        std_sensitivity = np.std(sensitivities)
        results.append(f'{mean_sensitivity:.3f} ± {std_sensitivity:.3f}')

        mean_specificity = np.mean(specificities)
        std_specificity = np.std(specificities)
        results.append(f'{mean_specificity:.3f} ± {std_specificity:.3f}')

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        results.append(f'{mean_auc:.3f} ± {std_auc:.3f}')

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        results.append(f'{mean_f1:.3f} ± {std_f1:.3f}')

        # finds the next fully NaN row
        next_row_index = df_output[df_output.isna().all(axis=1)].index[0]
        # Add the list to the row
        df_output.loc[next_row_index] = results
        results = []

df_output.to_excel(r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\plots\no_dementia_treatment\model_results_no_dem_feature.xlsx")





