import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
cm = confusion_matrix(y_test, y_pred)

print("Random Forest Results:")
print(cm)
print(classification_report(y_test, y_pred))

# Plot
sns.heatmap(cm, annot=True, fmt='d')
plt.title("RF Confusion Matrix")
plt.show()