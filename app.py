from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
import pickle


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/#classifier', methods=['GET'])
def classifier():
    return render_template('index.html', _anchor="classifier")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_dict = request.form.to_dict()

    # convert data_dict back into dataframe for preprocessing
    data = pd.DataFrame(data_dict, index=[1])
    loaded_model = pickle.load(open("model.pkl","rb"))

    # preprocess raw categorical data
    data['num_lab_procedures'] = pd.to_numeric(data.num_lab_procedures)
    data['num_medications'] = pd.to_numeric(data.num_medications)
    data['time_in_hospital'] = pd.to_numeric(data.time_in_hospital)
    data['age'] = pd.to_numeric(data.age)
    data['number_diagnoses'] = pd.to_numeric(data.number_diagnoses)
    data['num_procedures'] = pd.to_numeric(data.num_procedures)
    data['number_inpatient'] = pd.to_numeric(data.number_inpatient)
    data['number_outpatient'] = pd.to_numeric(data.number_outpatient)
    data['number_emergency'] = pd.to_numeric(data.number_emergency)

    data['change'] = data.loc[:, ('change')].replace('No', 0)
    data['change'] = data.loc[:, ('change')].replace('Ch', 1)

    data['insulin'] = data['insulin'].replace(['Steady', 'Up', 'Down'], 1)
    data['insulin'] = data['insulin'].replace('No', 0)

    data['metformin'] = data['metformin'].replace(['Steady', 'Up', 'Down'], 1)
    data['metformin'] = data['metformin'].replace('No', 0)

    data['gender'] = data['gender'].replace('Male', 0)
    data['gender'] = data['gender'].replace('Female', 1)
    data['Female'] = data['gender']
    data.drop(['gender'], axis=1, inplace=True)


    def function_circulatory(x):
        if x == 'circulatory':
            return 1
        else:
            return 0

    data['circulatory'] = data['diag_2'].apply(function_circulatory)

    # creating new column that groups diseases related to diabetes
    related_diseases = ['diabetes', 'musculoskeletal','thyroid',
                        'skin', 'nutritional, metabolic, immunity']

    def not_related(x):
        if x in related_diseases:
            return 0
        else:
            return 1

    data['not_diabetes_related'] = data['diag_2'].apply(not_related)

    data.drop(['diag_2'], axis=1, inplace=True)
    data.columns = ['num_lab_procedures',
                    'num_medications', 'time_in_hospital',
                    'age', 'number_diagnoses',
                    'num_procedures', 'number_inpatient',
                    'number_outpatient', 'number_emergency',
                    'change', 'insulin', 'metformin',
                    'not_diabetes_related', 'circulatory',
                    'Female']

    prediction_proba = loaded_model.predict_proba(data)
    prediction = (prediction_proba[0])[1] * 100
    prediction = round(prediction, 2)

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)