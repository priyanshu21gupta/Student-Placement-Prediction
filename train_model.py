import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Reproducible results
np.random.seed(42)

# Number of students
n = 500

# Generate student data
data = pd.DataFrame({
    "Student_ID": [f"STU{i:03d}" for i in range(1, n + 1)],
    
    "CGPA": np.round(np.random.uniform(5.0, 9.8, n), 2),
    
    "Attendance": np.round(np.random.uniform(55, 100, n), 1),
    
    "Backlogs": np.random.randint(0, 5, n),
    
    "Coding_Skill": np.random.randint(1, 11, n),
    
    "Communication_Skill": np.random.randint(1, 11, n),
    
    "Projects": np.random.randint(0, 6, n),
    
    "Internships": np.random.randint(0, 3, n),
    
    "Aptitude_Score": np.random.randint(40, 101, n)
})

# Calculate placement probability
score = (
    data["CGPA"] * 10
    + data["Attendance"] * 0.25
    - data["Backlogs"] * 8
    + data["Coding_Skill"] * 4
    + data["Communication_Skill"] * 3
    + data["Projects"] * 4
    + data["Internships"] * 6
    + data["Aptitude_Score"] * 0.20
)

# Add some randomness
score += np.random.normal(0, 8, n)

# Convert score into placement
data["Placement"] = np.where(score >= 135, "Yes", "No")

# Create data directory
os.makedirs("data", exist_ok=True)

# Save dataset
data.to_csv("data/student_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total students: {len(data)}")

print("\nPlacement distribution:")
print(data["Placement"].value_counts())

# -----------------------------
# Machine Learning
# -----------------------------

features = [
    "CGPA",
    "Attendance",
    "Backlogs",
    "Coding_Skill",
    "Communication_Skill",
    "Projects",
    "Internships",
    "Aptitude_Score"
]

X = data[features]
y = data["Placement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/placement_model.pkl")

print("\nModel saved successfully!")
print("File: model/placement_model.pkl")