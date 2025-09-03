import cv2
import numpy as np

def extract_features_and_draw(image_path):
    features = {}
    img = cv2.imread(image_path)
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img

    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2)

    # Feature 1: Area
    area = cv2.contourArea(contour)
    features['area'] = area

    # Feature 2: Perimeter
    perimeter = cv2.arcLength(contour, True)
    features['perimeter'] = perimeter

    # Feature 3: Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    features['solidity'] = solidity

    # Feature 4: Corners
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features['num_corners'] = len(approx)

    # Feature 5: Curvature radius
    if len(contour) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        curvature = (MA + ma) / 2
    else:
        curvature = 0
    features['curvature_radius'] = curvature

    # Feature 6: Symmetry
    (h, w) = gray.shape
    mid_x = w // 2
    left = gray[:, :mid_x]
    right = gray[:, mid_x:]
    right_flipped = cv2.flip(right, 1)

    if left.shape != right_flipped.shape:
        min_h = min(left.shape[0], right_flipped.shape[0])
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:min_h, :min_w]
        right_flipped = right_flipped[:min_h, :min_w]

    diff = cv2.absdiff(left, right_flipped)
    symmetry_score = np.mean(diff)
    features['symmetry_score'] = symmetry_score

    # Draw features on image
    y_offset = 30
    for key, val in features.items():
        text = f"{key}: {val:.1f}"
        cv2.putText(output_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25

    return features, output_img

# Example test run
if __name__ == "__main__":
    img_path = "test.jpg"
    features, output = extract_features_and_draw(img_path)
    if features:
        cv2.imshow("Features", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No contour found.")
