import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_model(model_name):
    model_map = {
        "Random Forest": "models/fraud_detector.pkl",
        "Logistic Regression": "models/fraud_detector.pkl",
        "XGBoost": "models/fraud_detector.pkl"
    }
    return joblib.load(model_map[model_name])

def predict_transaction(model, data):
    prediction = model.predict(data.values)
    return prediction[0]

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Feature Importance")
    return fig
