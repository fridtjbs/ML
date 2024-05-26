# generate binary classification dataset and plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.datasets import make_blobs
# generate dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)

X_df = pd.DataFrame(X, columns=['x1','x2'])
print(X_df.head())
print(X)
print(y)
