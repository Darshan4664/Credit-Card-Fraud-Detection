import pandas as pd

def get_top_suspicious_transactions(df, model, n=5):
    X = df.drop(columns=['Class'])
    preds = model.predict_proba(X)[:, 1]  # Get fraud probability
    df_copy = df.copy()
    df_copy['Fraud_Prob'] = preds
    top_suspicious = df_copy.sort_values(by='Fraud_Prob', ascending=False).head(n)
    return top_suspicious[['Fraud_Prob'] + list(X.columns) + ['Class']]
