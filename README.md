# J.P. Morgan Quant: Loan Default & Expected Loss Model 

## Project Overview
This project is an implementation of a quantitative credit risk model based on the J.P. Morgan Chase Quantitative Research task. The primary objective is to estimate the **Expected Loss (EL)** of a loan portfolio by predicting the **Probability of Default (PD)** for individual borrowers using machine learning techniques.

## Core Financial Logic
In credit risk modeling, the Expected Loss is calculated using the standard formula:
**EL = PD × EAD × LGD**

* **PD (Probability of Default):** Predicted using a Logistic Regression model based on borrower features.
* **EAD (Exposure at Default):** The outstanding loan amount (`loan_amt_outstanding`).
* **LGD (Loss Given Default):** Assumed to be 0.9 (based on a standard 10% recovery rate).

## Key Features & Methodology
1.  **Feature Engineering:** * Constructed key financial ratios to better capture borrower risk profiles:
        * `payment_to_income` = Loan Amount / Income
        * `debt_to_income` = Total Debt / Income
2.  **Data Preprocessing:** * Applied `StandardScaler` to normalize feature distributions, ensuring optimal convergence and performance of the Logistic Regression model.
3.  **Model Training:** * Trained a `LogisticRegression` classifier to output probabilities (PD) rather than simple binary classifications.
4.  **Portfolio Aggregation:** * Calculated the individual Expected Loss for each borrower and aggregated them to find the total expected loss for the entire loan portfolio.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (LogisticRegression, StandardScaler)

## File Structure
* `main.py`: The core script containing data processing, model training, and expected loss calculation logic.
* `Task 3 and 4_Loan_Data.csv`: The dataset containing borrower credit lines, outstanding debt, income, years employed, and FICO scores. (Note: Data used is for educational/task purposes).
