import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

# Load dataset
data = fetch_openml('adult', version=2, as_frame=True)
df = data.frame.copy()

# Prepare target and features
y = df['class'].map({'<=50K': 0, '>50K': 1})
X = df.drop(columns=['class'])

# Store sensitive feature for fairness evaluation
sensitive = X['sex']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, stratify=y, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Fairness evaluation
metric_frame = MetricFrame(
    metrics={
        'Accuracy': accuracy_score,
        'Selection Rate': selection_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=s_test
)

# Demographic parity difference
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
metric_frame.by_group.plot.bar(ax=axes[0])
axes[0].set_title("Fairness Metrics by Gender")
axes[0].legend(loc="best")
axes[0].grid(True)

# Feature importance
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
sns.barplot(data=coef_df.head(10), x='Coefficient', y='Feature', ax=axes[1])
axes[1].set_title("Top 10 Influential Features")
axes[1].grid(True)

plt.suptitle("Fairness & Feature Impact")
plt.tight_layout()
plt.show()

# Console Output
print("=== Group-wise Fairness Metrics ===")
print(metric_frame.by_group)
print(f"\nDemographic Parity Difference: {dpd:.3f}")
