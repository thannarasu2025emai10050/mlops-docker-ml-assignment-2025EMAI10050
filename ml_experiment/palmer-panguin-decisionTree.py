import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

# Load dataset
penguins = fetch_openml(name="penguins", version=1, as_frame=True)
df = penguins.frame

# Drop missing values
df = df.dropna()

# Encode target
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

X = df.drop('species', axis=1)
y = df['species']

# Encode categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Metrics
precision = precision_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print("Precision:", precision)
print("AUC Score:", auc)
