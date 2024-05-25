import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

# Load the Iris dataset
traindata = pd.read_csv("hw4Train.csv")
testdata = pd.read_csv("hw4Test.csv")
print(testdata.head())
print(traindata.head())

X = traindata.iloc[:, :-1]  # All columns except the last
y = traindata.iloc[:, -1]   # The last column


feature_names = X.columns

# Scale features to a range [0,1] for chi-square calculation
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Information Gain (using mutual information)
info_gain = mutual_info_classif(X, y)
info_gain_series = pd.Series(info_gain, index=feature_names)
info_gain_ranking = info_gain_series.sort_values(ascending=False)

# Chi-square test
chi2_values, p_values = chi2(X_scaled, y)
chi2_series = pd.Series(chi2_values, index=feature_names)
chi2_ranking = chi2_series.sort_values(ascending=False)

# Display the rankings
print("Feature ranking based on Information Gain:")
print(info_gain_ranking)

print("\nFeature ranking based on Chi-square:")
print(chi2_ranking)
