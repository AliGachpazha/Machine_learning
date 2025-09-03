import cv2
import numpy as np
import joblib

def curvature_radius(p1, p2, p3):
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-6))  # جلوگیری از sqrt منفی

    if area == 0:
        return float('inf')  # نقاط روی خط راست

    radius = (a * b * c) / (4.0 * area)
    return radius

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5), img, None, 0

    c = max(contours, key=cv2.contourArea)

    # Features
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
    corners = len(approx)

    # Curvature estimation
    contour = c.reshape(-1, 2)
    radii = []
    for i in range(1, len(contour) - 1):
        p1, p2, p3 = contour[i - 1], contour[i], contour[i + 1]
        radius = curvature_radius(p1, p2, p3)
        if radius < 10000:  # حذف مقادیر غیرواقعی
            radii.append(radius)

    mean_radius = np.mean(radii) if radii else 0

    return [area, perimeter, solidity, corners, hull_area], img, c, mean_radius

# Load trained model
model = joblib.load('model.pkl')

# Test image path
test_image_path = 'test.jpg'

# Extract features and prediction
features, gray_img, contour, mean_radius = extract_features(test_image_path)
prediction = model.predict([features])[0]

# Label and color
if prediction == 0:
    label = "Healthy ✅"
    color = (0, 255, 0)
else:
    label = "Defective ❌"
    color = (0, 0, 255)

# Convert image to BGR
output_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# Draw contour
if contour is not None:
    cv2.drawContours(output_img, [contour], -1, color, 2)

# Draw labels
cv2.putText(output_img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
cv2.putText(output_img, f"Curvature radius: {mean_radius:.1f} px", (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

# Show result
cv2.imshow("Prediction Result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save result image
cv2.imwrite("output_result.jpg", output_img)
