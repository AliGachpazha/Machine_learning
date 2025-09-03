# Defect Detection Project with Image Processing and Machine Learning

## 📁 Project Structure
```
project/
├── test_data/
│   ├── healthy/      # Images of healthy components
│   └── defect/       # Images of defective components
├── model.pkl         # Trained model
├── main.py           # Main project file
└── README.md         # This file
```

## 🔍 Image Processing

This project uses image processing techniques to extract key features from images:

1. **Grayscale Conversion**: Reduces data dimensions and focuses on structural features
2. **Gaussian Filter**: Removes noise and smooths the image
3. **Edge Detection with Canny Algorithm**: Identifies object boundaries in the image
4. **Contour Finding**: Extracts boundary points of the main object
5. **Geometric Feature Extraction**:
   - Area: Overall size of the object
   - Perimeter: Length of the object's boundary
   - Solidity: Ratio of area to convex hull area
   - Corner Count: Number of points in the polygonal approximation
   - Hull Area: Area of the smallest convex region that contains the object

## 🚀 Project Setup

### Prerequisites
- Python 3.7 or higher
- Required libraries

### Install Required Libraries

```bash
pip install -r requierments.txt
```

### Data Preparation

1. Create a `data` folder in the main project directory
2. Inside the `data` folder, create two subfolders: `healthy` and `defect`
3. Place images of healthy components in the `healthy` folder
4. Place images of defective components in the `defect` folder

### Running the Project

```bash
python main.py
```

### Execution Output

After running the program:
1. The model is trained on the training data
2. The trained model is saved in the `model.pkl` file
3. The model evaluation report on test data is displayed

## 📊 Model Performance

The model uses the RandomForestClassifier algorithm for classification which:
- Provides high accuracy on small-dimensional data
- Is resistant to overfitting
- Allows feature importance extraction

The classification report in the output shows precision, recall, and F1-score for each class.

## 🔮 Future Development

- Adding more features from images
- Using deep neural networks
- Creating a graphical user interface
- Developing as a web service

---

For any questions or issues regarding project setup, please create an issue in the project repository.