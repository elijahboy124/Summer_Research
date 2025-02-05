import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\preprocessing\tables.xlsx"
sheets_dict = pd.read_excel(file_path, sheet_name=None)

# Access individual sheets by their names
ctdata_df = sheets_dict["CTData"]
master_df = sheets_dict["Master"]
inpatient_df = sheets_dict["Inpatient"]
ed_df = sheets_dict["ED"]
outpatient_df = sheets_dict["Outpatient"]
contact_df = sheets_dict["Contact"]
delirium_df = sheets_dict["Delirium"]
blood_df = sheets_dict["BloodTests"]
pharmacy_df = sheets_dict["Pharmacy"]
arc_df = sheets_dict["ARC"]

# initialise the data dataframe
data = master_df[['id_patient', 'dementia']]
data['dementia'] = data['dementia'].fillna(0)


##
# CTData
##

# merge the id_patient into the ct data using the databse_number in master
ctdata_df = ctdata_df.merge(master_df[["id_patient", "database_number"]], on="database_number", how="left")
# merge ct data into data dataframe
data = data.merge(ctdata_df[["id_patient", "TBV", "TIV", "TBV/TIV", "L_HC", "R_HC", "Combined_HC", "HC/TIV"]], on="id_patient", how="left")


##
# Inpatient
##

# access the total length of stay information
inpatient_df["arrived"] = pd.to_datetime(inpatient_df["admission_datetime"])
inpatient_df["departed"] = pd.to_datetime(inpatient_df["discharge_datetime"])
inpatient_df["length"] = inpatient_df['departed'] - inpatient_df['arrived']
# filter for unique patient encounter
inpatient_df = inpatient_df.drop_duplicates(subset="inpatient_encounter_number", keep="first")
# sum total encounters and length of total encounters
total_durations = inpatient_df.groupby("id_patient")["length"].sum()
admission_counts = inpatient_df.groupby("id_patient").size()
# merge info into main dataframe
data = data.merge(admission_counts.rename("Number of Admissions - Inpatient"), on="id_patient", how="left")
data = data.merge(total_durations.rename("Total Length of Stay - Inpatient"), on="id_patient", how="left")

data['Number of Admissions - Inpatient'] = data['Number of Admissions - Inpatient'].fillna(0)
data['Total Length of Stay - Inpatient'] = data['Total Length of Stay - Inpatient'].fillna(0)


##
# ED
##

# count number of admissions to ED 
admission_counts = ed_df.groupby("id_patient").size()
data = data.merge(admission_counts.rename("Number of Admissions- ED"), on="id_patient", how="left")
data['Number of Admissions- ED'] = data['Number of Admissions- ED'].fillna(0)

##
# Outpatient
##

# get unique lists
data_outpatient = master_df[['id_patient']]
specialty = outpatient_df["outpatient_specialty_desc"].drop_duplicates()
attendance = outpatient_df["attendance_status"].drop_duplicates()

# loop through types of attendance and count how many each patient had
for attend in attendance:
    temp = outpatient_df[outpatient_df["attendance_status"] == attend]
    temp_count = temp.groupby("id_patient").size()
    data_outpatient = data_outpatient.merge(temp_count.rename(f'{attend} - Outpatient'), on="id_patient", how="left")
    data_outpatient[f'{attend} - Outpatient'] = data_outpatient[f'{attend} - Outpatient'].fillna(0)

# loop through specialties and count how many appointments per specialty
for spec in specialty:
    temp = outpatient_df[outpatient_df["outpatient_specialty_desc"] == spec]
    temp_count = temp.groupby("id_patient").size()
    data_outpatient = data_outpatient.merge(temp_count.rename(f'{spec} - Outpatient'), on="id_patient", how="left")
    data_outpatient[f'{spec} - Outpatient'] = data_outpatient[f'{spec} - Outpatient'].fillna(0)


# data_outpatient.to_excel("data_outpatient.xlsx", index=False)


##
# Contact
##

# count the number of cancelled, occured and planned contacts
canc = contact_df[contact_df["contact_status_desc"] == "Cancelled"].groupby("id_patient").size()
occur = contact_df[contact_df["contact_status_desc"] == "Occurred"].groupby("id_patient").size()
planned = contact_df[contact_df["contact_status_desc"] == "Planned"].groupby("id_patient").size()

data = data.merge(canc.rename("Cancelled - Contact"), on="id_patient", how="left")
data = data.merge(occur.rename("Occurred - Contact"), on="id_patient", how="left")
data = data.merge(planned.rename("Planned - Contact"), on="id_patient", how="left")

data['Cancelled - Contact'] = data['Cancelled - Contact'].fillna(0)
data['Occurred - Contact'] = data['Occurred - Contact'].fillna(0)
data['Planned - Contact'] = data['Planned - Contact'].fillna(0)

##
# Delirium
##

# total number of cams administered
count_cam = delirium_df.groupby("id_patient").size()
data = data.merge(count_cam.rename("Number of CAMS - Delirium"), on="id_patient", how="left")

# filter for id and admission time to get total number of admissions with CAM performed
del_admissions = delirium_df.drop_duplicates(subset=["id_patient", "admission_datetime"])
count_admissions = del_admissions.groupby("id_patient").size()
data = data.merge(count_admissions.rename("Number of Admissions - Delirium"), on="id_patient", how="left")

# get total number of delirium positive cases
delirium_df['positive'] = ((delirium_df["total_score"] >= 3) & (delirium_df["CAM_Score_3._Evidence_Focus"] == "Yes")).astype(int)
positive_del = delirium_df.groupby("id_patient")["positive"].sum()
data = data.merge(positive_del.rename("Number of Positive - Delirium"), on="id_patient", how="left")

data['Number of CAMS - Delirium'] = data['Number of CAMS - Delirium'].fillna(0)
data['Number of Admissions - Delirium'] = data['Number of Admissions - Delirium'].fillna(0)
data['Number of Positive - Delirium'] = data['Number of Positive - Delirium'].fillna(0)



##
# Blodtests
##

# count number of blood tests
test = blood_df.groupby("id_patient").size()
data = data.merge(test.rename("Number of Blood Tests - BloodTests"), on="id_patient", how="left")
# filter for abnormal test result and count
blood_norm = blood_df[blood_df['test_result_abnormal'] != 'N']
abnormal = blood_norm.groupby("id_patient").size()
data = data.merge(abnormal.rename("Number of Abnormal Tests - BloodTests"), on="id_patient", how="left")

data['Number of Blood Tests - BloodTests'] = data['Number of Blood Tests - BloodTests'].fillna(0)
data['Number of Abnormal Tests - BloodTests'] = data['Number of Abnormal Tests - BloodTests'].fillna(0)


# getting result of most recent blood test for each patient
blood_df["test_result_datetime"] = pd.to_datetime(blood_df["test_result_datetime"])
blood_df = blood_df.sort_values(by='test_result_datetime', ascending=False)
blood_df["test_result"] = pd.to_numeric(blood_df["test_result"], errors='coerce')
blood_df['test_name'] = blood_df['test_name'] + ' - BloodTests'

# Drop duplicates to keep only the most recent test result for each patient-test combination
recent_blood = blood_df.drop_duplicates(subset=['id_patient', 'test_name'], keep='first')
results_df = recent_blood.pivot(index='id_patient', columns='test_name', values='test_result')
results_df = results_df.reset_index()
#results_df.to_excel("data_blood.xlsx", index=False)

for col in results_df.columns:
    if results_df[col].isna().sum() > 8:
        results_df.drop(columns=[col], inplace=True)
#results_df.to_excel("data_blood_new.xlsx", index=False)
data = data.merge(results_df, on='id_patient', how='left')


##
# Pharmacy
##

# unique list of therapeutic groups
data_pharmacy = master_df[['id_patient']]
tg = pharmacy_df["TG2"].drop_duplicates()

# loops through TG and counts the number of prescriptions for each patient
for drug in tg:
    temp = pharmacy_df[pharmacy_df["TG2"] == drug]
    temp_count = temp.groupby("id_patient").size()
    data_pharmacy = data_pharmacy.merge(temp_count.rename(f'{drug} - Pharmacy'), on="id_patient", how="left")
    data_pharmacy[f'{drug} - Pharmacy'] = data_pharmacy[f'{drug} - Pharmacy'].fillna(0)


#data_pharmacy.to_excel("data_pharmacy.xlsx", index=False)


##
# ARC
##

dementia_age = arc_df[arc_df["Service.Category"] == "DEMENTIA-AGE"].groupby("id_patient")["No..of.Units"].sum()
hospital_age = arc_df[arc_df["Service.Category"] == "HOSPITAL-AGE"].groupby("id_patient")["No..of.Units"].sum()
psycgeri_age = arc_df[arc_df["Service.Category"] == "PSYCGERI-AGE"].groupby("id_patient")["No..of.Units"].sum()
resthome_age = arc_df[arc_df["Service.Category"] == "RESTHOME-AGE"].groupby("id_patient")["No..of.Units"].sum()

data = data.merge(dementia_age.rename("Dementia Stay Days - ARC"), on="id_patient", how="left")
data = data.merge(hospital_age.rename("Hospital Stay Days - ARC"), on="id_patient", how="left")
data = data.merge(psycgeri_age.rename("Psycgeri Stay Days - ARC"), on="id_patient", how="left")
data = data.merge(resthome_age.rename("Resthome Stay Days - ARC"), on="id_patient", how="left")

data['Dementia Stay Days - ARC'] = data['Dementia Stay Days - ARC'].fillna(0)
data['Hospital Stay Days - ARC'] = data['Hospital Stay Days - ARC'].fillna(0)
data['Psycgeri Stay Days - ARC'] = data['Psycgeri Stay Days - ARC'].fillna(0)
data['Resthome Stay Days - ARC'] = data['Resthome Stay Days - ARC'].fillna(0)



#data.to_excel("data.xlsx", index=False)

data = data.merge(data_outpatient, on="id_patient", how="left")
data = data.merge(data_pharmacy, on="id_patient", how="left")

data.to_csv(r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final.csv", index=False)
data.to_excel(r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final.xlsx", index=False)

data_cleaned = data.dropna()
data_cleaned.to_excel(r"C:\Users\ehay055\OneDrive - The University of Auckland\Desktop\research\model\data_final_cleaned.xlsx", index=False)

