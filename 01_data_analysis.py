import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')

# Display basic info
df.info()
print('\nFirst 5 rows:')
print(df.head())

# Churn rate and imbalance
churn_rate = df['Churn'].value_counts(normalize=True)
print('\nChurn rate:')
print(churn_rate)

# Plot churn distribution
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Correlation heatmap (for numeric features)
numeric_features = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_features.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()

# List categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print('\nCategorical features:')
print(categorical_features)

# Show value counts for key categorical features
for col in ['Contract', 'PaymentMethod']:
    print(f'\nValue counts for {col}:')
    print(df[col].value_counts())
