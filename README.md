# Telco Customer Churn Prediction Project

This repository contains all relevant files for the churn prediction project using three different machine learning models: Random Forest, Artificial Neural Network (ANN), and XGBoost. The objective is to compare these models in predicting customer churn using the Telco Customer Churn dataset.

---
## Accessing and running the code:
1. First we need to restore a conda environment:
`conda env create -f area_51_cis579.yml`

2. After that, activate the env:
`conda activate area_51_cis579`
### --- OR ---
1. If not using conda, you can use the requirements.txt in the base folder, titled as requirements.txt. But make sure you have a new virtual environment active to avoid issues with existing packages:
`pip install -r requirements.txt`

---

##  Folder and File Descriptions

###  `webapp/`
Contains the code and assets required to deploy a web-based application for making churn predictions using the trained models.

## To run the streamlit webapp:
1. cd into the submission directory
2. clone the environments or install the requirements as suggested above
3. Run `ollama serve`
4. make sure you have gemma3:4b installed and ready. If not, run `ollama run gemma3:4b` and it will install the llm. Installing the LLM is not vital to run the streamlit interface, so this is totally optional if the LLM functionality is not of concern.
5. in the base directory, where you are out of the webapp directory, run `streamlit run webapp/main.py`
6. The webapp should run, but since we have enough try catch blocks, the webapp should tell you exactly where there is an issue, if there is, which there won't be, but just in case.

---

###  `ann.ipynb`
This Jupyter notebook includes the implementation of an **Artificial Neural Network (ANN)** model for churn prediction. It covers:
- Data preprocessing (specific to ANN input)
- Model architecture and training
- Evaluation metrics (accuracy, recall, F1-score, confusion matrix)

---

###  `box_scatterplots.ipynb`
Evaluates the preprocessed dataset by creating boxplots and scatterplots of some of the attributes against each other.

---

###  `CustomerChurn.xlsx`
Contains the original data downloaded from https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

###  `EDA_v2_with_plots.ipynb`

This notebook is dedicated to creating visualizations for:
- Feature distributions
- Class imbalance (e.g., churn vs no churn)
- Feature importances from Random Forest
- Correlation heatmaps and histograms

---

###  `preprocessed.csv`
A Microsoft Excel file containing the **cleaned and preprocessed version of the Telco dataset**, ready for model input. All categorical features have been encoded, and missing values handled.

---

###  `preprocessing.ipynb`
A standalone notebook responsible for:
- Loading the raw dataset
- Performing feature engineering
- Handling missing values
- Encoding categorical variables
- Exporting the cleaned dataset to `preprocessed.xlsx`

---

###  `rfc_smote.ipynb`
Implements the **Random Forest Classifier** with and without **SMOTE (Synthetic Minority Oversampling Technique)**. This notebook includes:
- Baseline Random Forest implementation
- Hyperparameter tuning with GridSearchCV
- Handling class imbalance
- Performance evaluation

---

###  `xgb.ipynb`
This notebook covers the **XGBoost model** implementation. It includes:
- Model training and evaluation
- Hyperparameter tuning
- Comparative analysis with other models

---

##  Note
Ensure that the `preprocessed.csv` file is available before running any of the modeling notebooks, as it serves as the data source for training.

---
##  Contact
For questions or contributions, please reach out to the project team.
