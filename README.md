Credit Card Fraud Detection:

A machine learning project that detects fraudulent credit card transactions using a real Kaggle dataset.
It includes data preprocessing, model training (Logistic Regression, Random Forest, XGBoost),data 
balancing using SMOTE, and a Streamlit web interface to test predictions


Features: >
  - Cleans and processes real-world credit card transaction data
  - Applies SMOTE to handle imbalanced class distribution
  - Trains three models and selects the best one based on accuracy
  - Saves trained model as a .pkl file
  - Streamlit frontend for model selection, fraud prediction, and accuracy display


Technologies_used: >
  - Python 3.7+
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - imbalanced-learn
  - streamlit
  - matplotlib
  - seaborn
  - joblib


Dataset Info: >

This project uses the Kaggle Credit Card Fraud dataset:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- 284,807 transactions
- Only 492 are frauds
- Features are anonymized (V1 to V28), plus Time and Amount
- Class = 1 means fraud, 0 means legit

You need to download this dataset from Kaggle and place the CSV file in the `data/` folder like this:

/data/creditcard.csv


Usage_instructions: >
  1. Install required packages with: pip install -r requirements.txt
  2. Train the model by running: notebooks/model_training.ipynb
  3. Launch the app using: streamlit run app/app.py
  4. Access the app in your browser at: http://localhost:8501
                                        There you can:
                                        - Choose a model
                                        - Select a transaction
                                        - Predict if it’s fraud
                                        - Compare prediction to actual


Notes: >
  - The dataset and model files are excluded via .gitignore.
  - You can retrain the model at any time using the notebook.
  - Accuracy is shown in the sidebar of the Streamlit app.


Author: >

Darshan Sonara  
Credit Card Fraud Detection with Python + Machine Learning + Streamlit
This project was completed during my 5th semester of college.

