# Customer Churn Prediction

## Business Context
A telecommunications company faces high customer churn, which is costly. Retaining customers is five times cheaper than acquiring new ones. This project builds a machine learning model to identify customers at high risk of churning, enabling targeted retention campaigns.

## Data Source
- Telco Customer Churn dataset from Kaggle

## Analysis & Preprocessing

# Customer Churn Prediction

## Business Context
A telecommunications company is experiencing high customer churn, which is costly. Retaining existing customers is five times less expensive than acquiring new ones. This project aims to build a machine learning model to identify customers at high risk of churning, enabling targeted retention campaigns.

## Data Source
- Telco Customer Churn dataset from Kaggle

## Analysis & Preprocessing
- **Churn Rate:** 26.54%
- **Class Imbalance:** The dataset is imbalanced, with a lower proportion of churned customers. Techniques such as SMOTE and class weights were used to address this.
- **Feature Correlation:** The most correlated features with churn are tenure, contract type, payment method, and monthly charges.
- **Categorical Features:** Categorical features such as contract type and payment method were encoded using one-hot encoding for modeling.

## Modeling
- **Models Used:**
  - Logistic Regression
  - Random Forest
- **Train/Test Split:** 80/20 split between training and testing sets.
- **Imbalance Handling:** SMOTE was used to balance the training data.

## Evaluation
- **Confusion Matrix:**
  See classification report below for details.

- **Classification Report:**

Random Forest:
              precision    recall  f1-score   support

           0       0.83      0.88      0.85      1035
           1       0.59      0.49      0.53       374

    accuracy                           0.77      1409
   macro avg       0.71      0.68      0.69      1409
weighted avg       0.76      0.77      0.77      1409

Logistic Regression:
              precision    recall  f1-score   support

           0       0.91      0.71      0.80      1035
           1       0.50      0.80      0.62       374

    accuracy                           0.74      1409
   macro avg       0.70      0.76      0.71      1409
weighted avg       0.80      0.74      0.75      1409

- **Precision vs. Recall:**
Recall is more important in this context, because losing a customer (false negative) is much more costly than targeting a customer who would not have left (false positive).

- **Feature Importances:**
num__TotalCharges                   0.129750
num__Tenure                         0.118448
num__MonthlyCharges                 0.114337
cat__Contract_Two year              0.067675
cat__PaymentMethod_Manual           0.067108
cat__PaperlessBilling_Yes           0.064486
cat__InternetService_Fiber optic    0.056572
cat__OnlineSecurity_Yes             0.044684
cat__Contract_One year              0.040986
cat__TechSupport_Yes                0.039509

## Business Insights & Recommendations
Based on the most important features, here are three actionable recommendations for the marketing team:

1. **Focus on month-to-month contracts:** Customers with month-to-month contracts are at the highest risk of churning. Offer incentives to migrate to longer-term contracts.
2. **Address payment method preferences:** Customers using manual or electronic check payments show higher churn rates. Provide benefits for switching to automatic payments or credit cards.
3. **Engage new customers:** Customers with low tenure are more likely to churn. Implement onboarding campaigns and early engagement strategies to increase retention.

## Conclusion
The best model achieved an F1-score of 0.77. By identifying high-risk customers, the company can reduce churn and save on acquisition costs.

## How to Run the Interactive Dashboard

1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
2. Run the dashboard:
  ```bash
  streamlit run dashboard.py
  ```
3. Access the dashboard in your browser at the provided local URL.

