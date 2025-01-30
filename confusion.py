import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('train.csv')  # Replace with the correct file path if needed
test_data = pd.read_csv('test.csv')   # Replace with the correct file path if needed

# Separate features and target variable
X_train = train_data.drop(columns=["fake"])  # Features from the training dataset
y_train = train_data["fake"]                # Target variable from the training dataset
X_test = test_data.drop(columns=["fake"])   # Features from the testing dataset
y_test = test_data["fake"]                  # True labels from the testing dataset

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap=plt.cm.Blues)

# Customize and show the plot
plt.title("Confusion Matrix")
plt.show()

# Print the confusion matrix values
print("Confusion Matrix:")
print(cm)
