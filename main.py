import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_expected_loss(csv_filepath):
    # Read CSV data
    df = pd.read_csv(csv_filepath)

    # Calculate the repayment-to-income ratio and the debt-to-income ratio
    df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
    df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

    # Calculate pd (probability of default)
    feature_cols = [
        'credit_lines_outstanding', 
        'debt_to_income', 
        'payment_to_income', 
        'years_employed', 
        'fico_score'
    ]
    features = df[feature_cols]
    target = df['default']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train a basic logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(features_scaled, target) 
    pd_values = model.predict_proba(features_scaled)[:, 1]

    # Calculate expected loss
    ead_values = df['loan_amt_outstanding']
    lgd = 0.9

    # Calculate each person's expected loss: pd * ead * lgd
    df['expected_loss'] = pd_values * ead_values * lgd

    # Add up the expected losses of all loans
    total_loss = df['expected_loss'].sum()
    final_result = int(round(total_loss))

    return final_result


final_loss_integer = calculate_expected_loss("Task 3 and 4_Loan_Data.csv")
print(final_loss_integer)