import cv2
import numpy as np

def detect_blade_edges(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    # Resize (اختیاری - اگر عکس خیلی بزرگه)
    img = cv2.resize(img, (800, 800))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Detected Blade Edges", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_blade_edges("test.jpg")
