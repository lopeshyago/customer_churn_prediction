import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")

# Load data
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

st.header("Data Overview")
st.dataframe(df.head())

# Churn rate
churn_counts = df['Churn'].value_counts()
churn_rate = churn_counts[1] / churn_counts.sum()
st.metric("Churn Rate", f"{churn_rate:.2%}")
fig_pie = px.pie(names=['No Churn', 'Churn'], values=churn_counts.values, title='Churn Rate')
st.plotly_chart(fig_pie, use_container_width=True)

# Feature correlations
st.header("Feature Correlations")
numeric_features = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_features.corr()
fig_corr = px.imshow(correlation, text_auto=True, title='Correlation Heatmap (Numeric Features)')
st.plotly_chart(fig_corr, use_container_width=True)

# Distributions
st.header("Feature Distributions")
fig_tenure = px.histogram(df, x='Tenure', color='Churn', barmode='overlay', title='Tenure Distribution by Churn')
st.plotly_chart(fig_tenure, use_container_width=True)
fig_monthly = px.histogram(df, x='MonthlyCharges', color='Churn', barmode='overlay', title='Monthly Charges Distribution by Churn')
st.plotly_chart(fig_monthly, use_container_width=True)

# Model training
st.header("Model Training & Evaluation")
X = df.drop('Churn', axis=1)
y = df['Churn']
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols),
    ('num', SimpleImputer(strategy='median'), X.select_dtypes(include=['float64', 'int64']).columns.tolist())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)
X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_bal, y_train_bal)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)

# Model results
y_pred_lr = lr.predict(X_test_enc)
y_pred_rf = rf.predict(X_test_enc)

st.subheader("Random Forest Classification Report")
st.text(classification_report(y_test, y_pred_rf))

st.subheader("Logistic Regression Classification Report")
st.text(classification_report(y_test, y_pred_lr))

# Feature importances
importances = rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
fig_imp = px.bar(feat_imp.head(10), title='Top 10 Feature Importances (Random Forest)')
st.plotly_chart(fig_imp, use_container_width=True)

# Business recommendations
st.header("Business Recommendations & Conclusion")
st.markdown("""
**Key Insights:**
- Customers with month-to-month contracts and manual/electronic check payments are at highest risk of churn.
- Tenure and monthly charges are strong predictors of churn.

**Recommendations:**
1. Target month-to-month contract customers with loyalty programs or incentives for longer-term contracts.
2. Encourage customers to switch from manual/electronic check payments to automatic payments or credit cards.
3. Engage new customers early with onboarding and retention campaigns.

**Conclusion:**
The machine learning models built here help identify high-risk customers, enabling the company to reduce churn and save on acquisition costs. This dashboard provides a clear, interactive view of the data and model results for recruiters and stakeholders.
""")
