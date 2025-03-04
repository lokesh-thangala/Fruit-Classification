from sklearn.tree import DecisionTreeClassifier
import numpy as np

data = np.array([[150, 1], [170, 1], [130, 0], [120, 0]])  # Weight, 1=Orange, 0=Apple
labels = ["Orange", "Orange", "Apple", "Apple"]
model = DecisionTreeClassifier().fit(data, labels)

print("Prediction:", model.predict([[140, 0]]))  # Example: Predicting an apple
