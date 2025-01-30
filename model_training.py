import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib

# Load Dataset
df = pd.read_csv('train.csv')

# Split features and target
X = df.drop('fake', axis=1)  # Assuming 'fake' is the target column
y = df['fake']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)

# Calculate Accuracy and Precision
accuracy = accuracy_score(y_test, y_pred)*100
precision = precision_score(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.0f}%")
print(f"Precision: {precision:.1f}")

# Save the model
joblib.dump(rf, 'fake_account_model.pkl')
