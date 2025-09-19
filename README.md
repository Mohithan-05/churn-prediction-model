🛠️ Churn Prediction Project
📌 Overview

This project predicts customer churn using machine learning models.
Churn refers to when users stop engaging with a service — for example, not booking, clicking, or searching for a period of time.

We trained multiple models (Logistic Regression, Random Forest, XGBoost) and used SMOTE to handle class imbalance.
The best-performing model (XGBoost) is then used to generate churn probabilities for each user.

🚀 Features

Preprocessing and feature engineering from multiple datasets (users.csv, bookings.csv, clicks.csv, search_logs.csv)

Creation of churn labels (users with no bookings in the last 60 days)

Handling of imbalanced data with SMOTE

Training and evaluation of 3 ML models:

Logistic Regression

Random Forest

XGBoost

Performance evaluation using ROC-AUC and Classification Reports

ROC Curve visualization (roc_curve.png)

Export of churned users to churned_users.csv

📂 Project Structure
churn_prediction_project/
│── main.py               # Main script for data processing, training, and evaluation
│── users.csv             # User information dataset
│── bookings.csv          # Bookings dataset
│── clicks.csv            # Click activity dataset
│── search_logs.csv       # Search activity dataset
│── churned_users.csv     # Output file: predicted churned users
│── roc_curve.png         # ROC Curve comparing models
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/churn_prediction_project.git
cd churn_prediction_project

2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Mac/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Usage
Run the project
python main.py

Outputs generated:

ROC Curve → roc_curve.png

Churned Users → churned_users.csv

📊 Example Output

ROC Curve comparing models:


Sample churned users file (churned_users.csv):

user_id,total_bookings,total_clicks,total_searches,churn_probability
102,5,12,3,0.87
156,2,8,1,0.76
198,0,3,0,0.94

📈 Models Used

Logistic Regression

Random Forest Classifier

XGBoost Classifier

📌 Next Steps (Improvements)

Hyperparameter tuning for better accuracy

Adding more behavioral features (support tickets, session activity, etc.)

Deploying the model with a Flask/FastAPI backend
