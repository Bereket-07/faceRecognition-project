import face_recognition as fr
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Data Collection
path = "./data/"
X = []
y = []

# Load images and labels
print("Loading data...")
for name in os.listdir(path):
    for image_name in os.listdir(os.path.join(path, name)):
        image = fr.load_image_file(os.path.join(path, name, image_name))
        encoding = fr.face_encodings(image)[0]  # Assuming only one face per image
        X.append(encoding)
        y.append(name)

# Step 2: Data Preprocessing
print("Data loaded.")

# Step 3: Feature Extraction (Already done during data collection)

# Step 4: Model Training
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("Model trained.")

# Step 5: Model Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


