import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = "test.jpg"  # change if needed
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150)

# Focus only on central area (e.g., 30% to 70% of image width)
h, w = edges.shape
x_start = int(w * 0.30)
x_end = int(w * 0.70)

# Zero out the left and right sides (remove background edges)
edges[:, :x_start] = 0
edges[:, x_end:] = 0

# For visualization
output = image.copy()
output[edges != 0] = [255, 0, 0]  # Show edges in blue

# Define vertical positions for diameter measurements
y_positions = [
    int(h * 0.25),  # top
    int(h * 0.5),   # middle
    int(h * 0.75)   # bottom
]

for y in y_positions:
    row = edges[y, :]
    x_points = np.where(row > 0)[0]
    if len(x_points) >= 2:
        x1, x2 = x_points[0], x_points[-1]
        length = x2 - x1
        cv2.line(output, (x1, y), (x2, y), (0, 255, 0), 2)
        cv2.putText(output, f"{length}px", (x1 + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show result
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))
plt.imshow(output_rgb)
plt.axis("off")
plt.title("Blade edge detection and diameter measurements (pixels)")
plt.show()
