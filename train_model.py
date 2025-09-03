import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)

    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
    corners = len(approx)

    return [area, perimeter, solidity, corners, hull_area]


healthy_dir = 'data/healthy'
defect_dir = 'data/defect'

X, y = [], []

for filename in os.listdir(healthy_dir):
    features = extract_features(os.path.join(healthy_dir, filename))
    X.append(features)
    y.append(0)

for filename in os.listdir(defect_dir):
    features = extract_features(os.path.join(defect_dir, filename))
    X.append(features)
    y.append(1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)


joblib.dump(model, 'model.pkl')


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
