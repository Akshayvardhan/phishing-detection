import pandas as pd

legitimate_train = pd.read_csv('features/legitimate_train.csv')
phish_train = pd.read_csv('features/phish_train.csv')

train = pd.concat([legitimate_train, phish_train], axis=0)

# Drop url to calculate correlation for numeric features
numeric_data = train.drop(columns=['url'])

correlation = numeric_data.corr()['result_flag'].sort_values(ascending=False)

print("Feature Correlation with Result Flag:")
print(correlation)
