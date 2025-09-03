import sys
import cv2
import numpy as np
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# Load trained model
model = joblib.load('model.pkl')

def curvature_radius(p1, p2, p3):
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-6))
    if area == 0:
        return float('inf')
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
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
    corners = len(approx)
    contour = c.reshape(-1, 2)
    radii = [curvature_radius(contour[i - 1], contour[i], contour[i + 1])
             for i in range(1, len(contour) - 1)
             if curvature_radius(contour[i - 1], contour[i], contour[i + 1]) < 10000]
    mean_radius = np.mean(radii) if radii else 0
    return [area, perimeter, solidity, corners, hull_area], img, c, mean_radius

class BladeHealthApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blade Health Classifier")
        self.showMaximized()
        self.setStyleSheet("background-color: #C0C0C0;")  # Silver background
        self.image_path = None

        # Layouts
        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.image_label = QLabel("Load an image to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.image_label.setStyleSheet("background-color: white;")

        # Buttons
        self.open_btn = QPushButton("Open Image")
        self.classify_btn = QPushButton("Classify")
        self.live_btn = QPushButton("Live Camera")
        self.save_btn = QPushButton("Save Result")
        self.clear_btn = QPushButton("Clear Image")
        self.exit_btn = QPushButton("Exit")

        for btn in [self.open_btn, self.classify_btn, self.live_btn, self.save_btn, self.clear_btn, self.exit_btn]:
            btn.setStyleSheet("font-size: 14px; padding: 6px 16px; background-color: red; color: black; font-weight: bold;")

        # Logo
        logo_label = QLabel()
        logo = QPixmap("logo.png")
        logo = logo.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo)
        logo_label.setAlignment(Qt.AlignHCenter)

        # Add widgets
        main_layout.addWidget(logo_label)
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.classify_btn)
        btn_layout.addWidget(self.live_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.image_label)
        self.setLayout(main_layout)

        # Connect signals
        self.open_btn.clicked.connect(self.open_image)
        self.classify_btn.clicked.connect(self.classify_image)
        self.save_btn.clicked.connect(self.save_result)
        self.clear_btn.clicked.connect(self.clear_image)
        self.exit_btn.clicked.connect(self.close)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.display_image(path)

    def display_image(self, path):
        pixmap = QPixmap(path).scaled(1000, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def classify_image(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "No image selected!")
            return
        features, gray_img, contour, mean_radius = extract_features(self.image_path)
        prediction = model.predict([features])[0]
        label = "Healthy" if prediction == 0 else "Defective"
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
        output = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        if contour is not None:
            cv2.drawContours(output, [contour], -1, color, 2)
        cv2.putText(output, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(output, f"Curvature radius: {mean_radius:.1f} px", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        height, width, channel = output_rgb.shape
        bytes_per_line = 3 * width
        qimg = QImage(output_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(1000, 600, Qt.KeepAspectRatio))
        self.last_result = output

    def save_result(self):
        if hasattr(self, 'last_result'):
            cv2.imwrite("classified_result.jpg", self.last_result)
            QMessageBox.information(self, "Saved", "Result saved as 'classified_result.jpg'")
        else:
            QMessageBox.warning(self, "Warning", "Nothing to save!")

    def clear_image(self):
        self.image_label.clear()
        self.image_label.setText("Load an image to begin")
        self.image_path = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BladeHealthApp()
    window.show()
    sys.exit(app.exec_())
