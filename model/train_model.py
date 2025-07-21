import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load data
df = pd.read_csv("data/sample_credit_data.csv")
features = ['monthly_topups', 'sms_sent',
            'payment_regular', 'digital_activity_score']
X = df[features]
y = df['label']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, probs))

# Save model
joblib.dump(model, "model/model.pkl")
