import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

users = pd.read_csv('users.csv')
bookings = pd.read_csv('bookings.csv')
clicks = pd.read_csv('clicks.csv')
search_logs = pd.read_csv('search_logs.csv')


bookings['booking_date'] = pd.to_datetime(bookings['booking_date'])


last_booking = bookings.groupby('user_id')['booking_date'].max().reset_index()
last_booking.columns = ['user_id', 'last_booking_date']
users = users.merge(last_booking, on='user_id', how='left')


reference_date = pd.to_datetime('2025-07-29')
users['days_since_last_booking'] = (reference_date - users['last_booking_date']).dt.days


users['churned'] = users['days_since_last_booking'].apply(lambda x: 1 if x > 60 else 0)

print("\nChurn value counts:")
print(users['churned'].value_counts())


booking_counts = bookings.groupby('user_id').size().reset_index(name='total_bookings')
click_counts = clicks.groupby('user_id').size().reset_index(name='total_clicks')
search_counts = search_logs.groupby('user_id').size().reset_index(name='total_searches')


users = users.merge(booking_counts, on='user_id', how='left')
users = users.merge(click_counts, on='user_id', how='left')
users = users.merge(search_counts, on='user_id', how='left')


users.fillna(0, inplace=True)


features = ['total_bookings', 'total_clicks', 'total_searches']
X = users[features]
y = users['churned']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}


for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")


plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()


print("\n📊 Predicting churn and exporting only churned users...")
final_model = models['XGBoost']
users['churn_probability'] = final_model.predict_proba(X[features])[:, 1]
users['churn_prediction'] = final_model.predict(X[features])


churned_users = users[users['churn_prediction'] == 1]
churned_users[['user_id', 'total_bookings', 'total_clicks', 'total_searches', 'churn_probability']].to_csv('churned_users.csv', index=False)
print("✅ Only churned users saved to churned_users.csv")


print("\nUnique values in y_test:", np.unique(y_test))
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} - Unique y_pred values:", np.unique(y_pred))


y_proba = final_model.predict_proba(X_test)[:, 1]
print(f"XGBoost - Min proba: {y_proba.min():.4f}, Max proba: {y_proba.max():.4f}")



print("Unique values in y_test:", np.unique(y_test))
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} - Unique y_pred values:", np.unique(y_pred))
print(f"{name} - Min proba: {y_proba.min()}, Max proba: {y_proba.max()}")







