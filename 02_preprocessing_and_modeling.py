import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')

# Drop customerID (not useful for modeling)
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric (handle errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
for col in ['TotalCharges']:
    df[col] = df[col].fillna(df[col].median())

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols),
    ('num', SimpleImputer(strategy='median'), X.select_dtypes(include=['float64', 'int64']).columns.tolist())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)
X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_bal, y_train_bal)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)

# Evaluate models
print('Logistic Regression Results:')
y_pred_lr = lr.predict(X_test_enc)
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print('Random Forest Results:')
y_pred_rf = rf.predict(X_test_enc)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Feature importances (Random Forest)
importances = rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print('\nTop 10 Feature Importances (Random Forest):')
print(feat_imp.head(10))

# Automate README update
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

def insert_results(readme, churn_rate, lr_report, rf_report, rf_features):
    import re
    # Fill churn rate
    readme = re.sub(r'(\*\*Churn Rate:\*\* _\[Preencher com o valor do seu script\]_)', f'**Churn Rate:** {churn_rate:.2%}', readme)
    # Fill classification report
    readme = re.sub(r'(\*\*Classification Report:\*\* _\[Inserir precision, recall, f1-score do melhor modelo\]_)', f'**Classification Report:**\n\nRandom Forest:\n{rf_report}\n\nLogistic Regression:\n{lr_report}', readme)
    # Fill feature importances
    readme = re.sub(r'(\*\*Feature Importances:\*\* _\[Listar principais features do Random Forest\]_)', f'**Feature Importances:**\n{rf_features}', readme)
    return readme

churn_rate_val = y.value_counts(normalize=True)[1]
lr_report_val = classification_report(y_test, y_pred_lr)
rf_report_val = classification_report(y_test, y_pred_rf)
rf_features_val = feat_imp.head(10).to_string()

new_readme = insert_results(readme, churn_rate_val, lr_report_val, rf_report_val, rf_features_val)
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(new_readme)
