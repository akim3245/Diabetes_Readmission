# Diabetes Readmission Classifier

Python Flask Web Application

### Methods Used
* Machine Learning
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* Pandas
* Scikit-Learn Library 
* Jupyter Notebook
* HTML
* CSS
* Flask
* Bootstrap

## Project Description
The goal of this project was to build a web app that can classify whether a patient will be readmitted after being 
discharged by using selective, key pieces of information. 

A diabetes dataset was used that consists of information from 130 US hospitals for years 1999-2008. There was a total of
 50 columns ranging from gender and race, to number of procedures and hospital visits, to various medication 
 prescriptions. 
 
Jupyter Notebook was used to run Python code. Data was first explored and analyzed to check for
trends and data visualizations (ie. matplotlib, plotly, sns) then preprocessed for clean-up and some tweaking and 
engineering to have it ready for machine learning. 

Overall, the dataset was relatively balanced across several features. The graphs below show a count plot of gender,
 patients prescribed with insulin, and whether patients were readmitted or not.

![Screenshot](gender.png?raw=true)

![Screenshot](insulin.png?raw=true)

![Screenshot](readmitted.png?raw=true)

Most readmitted patients were 50-90 years old.
Even within different age groups, there was a balanced distribution of patients that were readmitted and not readmitted.

![Screenshot](age_re.png?raw=true)

### Feature Engineering
Two columns were created to improve the model. 

The first new column was created to label the patient as Type1 or Type2 diabetic based on the medications they were prescribed.

The dataset contains 3 diagnoses columns 
(primary, secondary and tertiary) indicating disease type.
This information was used for the second new column which labels whether the disease is related to diabetes. 


### Modeling
Data was split then trained by several algorithms. XGBoost gave 
the best score and all the best parameters were found using Gridsearch. The top 15 features 
(out of 135 total predictive features) were extracted using SelectFromModel while maintaining the top ROC_AUC score. 


Top Features: 
* f2 - Number of Lab Procedures
* f4 - Number of Medications
* f1 - Time in Hospital (Length of Stay)
* f0 - Age
* f3 - Number of (non-lab) Procedures
* f8 - Number of Diagnoses
* f7 - Number of Inpatient Visits
* f5 - Number of Outpatient Visits
* f6 - Number of Emergency Visits
* f24 - Female
* f59 - Secondary Diagnosis being Circulatory
* 23 - Change in Medications
* f21 - Insulin
* f9 - Metformin
* f99 - Diagnosis/Disease is not Diabetes Related

![Screenshot](feature.png?raw=true)

### Results
To test the model, a random sample (shown below) from the holdout set was used 
to predict the probability of readmission status. 

![Screenshot](sample_data.png?raw=true)

The model was given the above sample information without the last line which shows the true readmission answer and produced 
the following result.

![Screenshot](test.png?raw=true)

The readmission status was predicted correctly for this sample with 
33.74% chance of being readmitted and 66.26% chance of not being readmitted.

Pre-trained data was deployed using Flask to make a diabetes readmission classifier web application to make predictions 
on user-submitted feature values. 


## Getting Started


#### 1. Create a Virtual Environment (optional but recommended)

   Install Anaconda here: https://www.anaconda.com/products/individual. 
   
   Once Anaconda is installed, create a new virtual environment called 
   ```new_env```, then switch out of the base environment and into ```new_env```. Also, make sure to use python3:
   
   ```
    $ conda create -n new_env python=3
    $ conda activate new_env
   ```
   
#### 2. Install Required Packages
   Use the **requirements.txt** file attached in this repository to install the necessary packages to run the app.


   ```
   (new_env)$ pip install -r requirements.txt
   ```
   
#### 3. Run the App
    
   ```
   $ python app.py
   ```
   Run the code above in your terminal and it should return 
   ```Running on http://127.0.0.1:5000/```.
   Copy the URL then paste it in a new browser and you should be able to see the web app.

    

