from flask import Flask, render_template, url_for, request, jsonify
from flask_bootstrap import Bootstrap
import pandas as pd
import pickle
import joblib


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict(data_dict):
    # convert data_dict back into dataframe for preprocessing
    data = pd.DataFrame(data_dict, index=[1])
    loaded_model = pickle.load(open("model.pkl","rb"))

    # preprocess raw categorical data
    data['insulin'] = data['insulin'].replace(['Steady', 'Up', 'Down'], 1)
    data['insulin'] = data['insulin'].replace('No', 0)

    data['change'] = data.loc[:, ('change')].replace('No', 0)
    data['change'] = data.loc[:, ('change')].replace('Ch', 1)

    for i in range(0, 10):
        data['age'] = data['age'].replace('[' + str(i * 10) + '-' + str(10 * (i + 1)) + ')', i + 1)

    data = data[~data['gender'].str.contains('Unknown/Invalid')]
    data['gender'] = data['gender'].replace('Male', 0)
    data['gender'] = data['gender'].replace('Female', 1)
    data['Female'] = data['gender']
    data.drop(['gender'], axis=1, inplace=True)

    data['metformin'] = data['metformin'].replace(['Steady', 'Up', 'Down'], 1)
    data['metformin'] = data['metformin'].replace('No', 0)

    data['diag_2'] = data['diag_2'].replace('?', 0)

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

    prediction = loaded_model.predict_proba(data)
    return prediction


@app.route('/result',methods=['POST'])
def result():
    # Receives the input query from form
    if request.method == 'POST':
        predict_user_input = request.form.to_dict()
        predict(predict_user_input)

        return render_template('result.html', prediction=prediction*100)


if __name__ == '__main__':
    app.run(debug=True)