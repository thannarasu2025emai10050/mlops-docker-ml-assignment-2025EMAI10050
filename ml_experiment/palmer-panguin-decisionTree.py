from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_auc_score
import pandas as pd

# Load Palmer Penguins dataset (using seaborn as proxy)
import seaborn as sns
df = sns.load_dataset('penguins').dropna()

# Prepare features
df['species'] = df['species'].astype('category').cat.codes
X = pd.get_dummies(df.drop('species', axis=1))
y = df['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Metrics
precision = precision_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print(f"Precision: {precision:.4f}")
print(f"AUC Score: {auc:.4f}")
