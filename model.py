from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load data
data = load_iris()
X, y = data.data, data.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")
