Diabetes Readmission Classifier
Python Flask Web Application

Methods Used
Machine Learning
Data Visualization
Predictive Modeling
Technologies
Python
Pandas
Scikit-learn library
Jupyter Notebook
HTML
CSS
Flask
Bootstrap
Project Description
The goal of this project was to build a web app that can classify whether a patient will be readmitted after being discharged by using selective, key pieces of information.

A diabetes dataset was used that consists of information from 130 US hospitals for years 1999-2008. There was a total of 50 columns ranging from gender and race, to number of procedures and hospital visits, to various medication prescriptions.

Jupyter Notebook was used to run Python code. Data was first explored and analyzed to check for trends and data visualizations (ie. matplotlib, plotly, sns), then preprocessed for clean-up and some tweaking and engineering to have it ready for machine learning. Data was split then trained by several algorithms. XGBoost gave the best score and was further used to extract the top 14 features. Pre-trained data was deployed using Flask to make the prediction probability of readmission.

Getting Started
#### 1. Create a Virtual Environment

Install Anaconda here: https://www.anaconda.com/products/individual.

Once Anaconda is installed, create a new virtual environment called new_env, then switch out of the base environment and into the new_env:

 $ conda create -n new_env python=3
 $ conda activate new_env
#### 2. Install Required Packages Use the requirements.txt file attached in this repository to install the necessary packages to run the app.

(new_env)$ pip install -r requirements.txt
#### 3. Run the App

$ python app.py
Run the code above in your terminal and it should return Running on http://127.0.0.1:5000/. Copy the URL then paste it in a new browser and you should be able to see the web app.