from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv('diabetic_data.csv')

features = ['age','num_medications','num_lab_procedures',
            'number_inpatient', 'number_outpatient','insulin',
            'time_in_hospital',
            'number_diagnoses', 'metformin','num_procedures',
            'change',
            'number_emergency', 'gender',
             'diag_2', 'circulatory',
            'not_diabetes_related''readmitted']

df = df.loc[:, features]

# preprocessing raw data
df['insulin'] = df['insulin'].replace(['Steady', 'Up', 'Down'], 1)
df['insulin'] = df['insulin'].replace('No', 0)

df['change'] = df.loc[:, ('change')].replace('No', 0)
df['change'] = df.loc[:, ('change')].replace('Ch', 1)

for i in range(0, 10):
    df['age'] = df['age'].replace('[' + str(i * 10) + '-' + str(10 * (i + 1)) + ')', i + 1)

df = df[~df['gender'].str.contains('Unknown/Invalid')]
gender_d = pd.get_dummies(df['gender'], prefix='gender')
df = pd.concat([df, gender_d], axis=1)
df['Female'] = df.gender_Female
df['gender'] = df['gender'].replace('Male', 0)
df['gender'] = df['gender'].replace('Female', 1)
df.drop(['gender', 'gender_Male', 'gender_Female'], axis=1, inplace=True)

df['metformin'] = df['metformin'].replace(['Steady', 'Up', 'Down'], 1)
df['metformin'] = df['metformin'].replace('No', 0)


def map_diag(diag_code):
    """
    Mapping diagnosis ID code to disease/disorder
    :param diag_code: number
    :return: category for a diagnosis
    """
    if "V" in str(diag_code) or "E" in str(diag_code):
        diag_category = 'external injury and supplemental'
    elif float(diag_code) is 0:
        diag_category = 'N/A'
    elif float(diag_code) < 140:
        diag_category = 'infectious and parasitic'
    elif float(diag_code) >= 140 and float(diag_code) < 240:
        diag_category = 'neoplasms'
    elif float(diag_code) >= 240 and float(diag_code) < 249:
        diag_category = 'thyroid'
    elif float(diag_code) >= 249 and float(diag_code) < 260:
        diag_category = 'diabetes'
    elif float(diag_code) >= 260 and float(diag_code) < 280:
        diag_category = 'nutritional, metabolic, immunity'
    elif float(diag_code) >= 280 and float(diag_code) < 290:
        diag_category = 'blood'
    elif float(diag_code) >= 290 and float(diag_code) < 320:
        diag_category = 'mental'
    elif float(diag_code) >= 320 and float(diag_code) < 390:
        diag_category = 'nervous'
    elif float(diag_code) >= 390 and float(diag_code) < 460:
        diag_category = 'circulatory'
    elif float(diag_code) >= 460 and float(diag_code) < 520:
        diag_category = 'respiratory'
    elif float(diag_code) >= 520 and float(diag_code) < 580:
        diag_category = 'digestive'
    elif float(diag_code) >= 580 and float(diag_code) < 630:
        diag_category = 'genitourinary'
    elif float(diag_code) >= 630 and float(diag_code) < 680:
        diag_category = 'pregnancy'
    elif float(diag_code) >= 680 and float(diag_code) < 710:
        diag_category = 'skin'
    elif float(diag_code) >= 710 and float(diag_code) < 740:
        diag_category = 'musculoskeletal'
    elif float(diag_code) >= 740 and float(diag_code) < 760:
        diag_category = 'congenital'
    elif float(diag_code) >= 760 and float(diag_code) < 780:
        diag_category = 'perinatal'
    elif float(diag_code) >= 780 and float(diag_code) < 800:
        diag_category = 'symptoms'
    else:
        diag_category = 'injury and poisoning'
    return diag_category


df['diag_2'] = df['diag_2'].replace('?', 0).apply(map_diag)


def function_circulatory(x):
    if x == 'circulatory':
        return 1
    else:
        return 0


df['circulatory'] = df['diag_2'].apply(function_circulatory)
diag_2_d = pd.get_dummies(df['diag_2'], prefix='diag2')
df = pd.concat([df, diag_2_d], axis=1)

df['readmitted'] = df.loc[:, ('readmitted')].replace('NO', 0)
df['readmitted'] = df.loc[:, ('readmitted')].replace(['>30', '<30'], 1)


# creating new column that groups diseases related to diabetes
def not_related(x):
    if x in related_diseases:
        return 0
    else:
        return 1


related_diseases = ['diabetes', 'musculoskeletal',
                    'thyroid', 'skin', 'nutritional, metabolic, immunity']

df['not_diabetes_related'] = df['diag_2'].apply(not_related)

df.drop(['diag2_blood', 'diag2_circulatory', 'diag2_congenital', 'diag2_diabetes', 'diag2_digestive',
         'diag2_external injury and supplemental', 'diag2_genitourinary', 'diag2_infectious and parasitic',
         'diag2_injury and poisoning', 'diag2_mental', 'diag2_musculoskeletal', 'diag2_neoplasms',
         'diag2_nervous', 'diag2_nutritional, metabolic, immunity', 'diag2_pregnancy', 'diag2_respiratory',
         'diag2_skin', 'diag2_symptoms', 'diag2_thyroid'], axis=1, inplace=True)

df.drop(['diag_2'], axis=1, inplace=True)

features = df.drop('readmitted', axis=1)
target = df.readmitted

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

xg = XGBClassifier(objective="binary:logistic", learning_rate=0.1, n_estimators=150, max_depth=8,
                   min_child_weight=3, gamma=0.2, subsample=0.9, colsample_bytree=0.7)
xg.fit(X_train, y_train)

# save model to file
pickle.dump(xg, open("model.pkl", "wb"))
