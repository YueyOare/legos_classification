# LEGO Brick Classification with Deep Learning

A computer vision project that uses Convolutional Neural Networks (CNNs) to classify different types of LEGO bricks from grayscale images.

## 📋 Project Overview

This project implements and compares two CNN architectures for classifying 10 different types of LEGO bricks:
- A simple custom CNN
- LeNet architecture

Both models achieve **100% accuracy** on the test dataset, demonstrating excellent performance for this classification task.

## 🧱 LEGO Brick Classes

The model can classify the following 10 types of LEGO pieces:

1. **2x3 Brick**
2. **2x2 Brick** 
3. **1x3 Brick**
4. **2x1 Brick**
5. **1x1 Brick**
6. **2x2 Macaroni**
7. **2x2 Curved End**
8. **Cog 16 Tooth**
9. **1x2 Handles**
10. **1x2 Grill**

## 📊 Dataset

- **Training samples**: 451 images
- **Test samples**: 150 images
- **Image dimensions**: 48x48 pixels (grayscale)
- **Data format**: Pickle files containing preprocessed image data

## 🏗️ Model Architectures

### Simple CNN
- 2 Convolutional layers (32 and 64 filters)
- 2 MaxPooling layers
- 1 Dense hidden layer (64 units)
- Output layer (10 classes)
- **Total parameters**: 429,130

### LeNet Architecture
- 2 Convolutional layers (6 and 16 filters with 5x5 kernels)
- 2 MaxPooling layers
- 2 Dense hidden layers (120 and 84 units)
- Output layer (10 classes)
- **Total parameters**: 205,706

## 🔧 Requirements

```python
torch
tensorflow
keras
matplotlib
seaborn
scikit-learn
numpy
visualkeras
pickle
```

## 🚀 Usage

1. **Install dependencies**:
   ```bash
   pip install torch tensorflow keras matplotlib seaborn scikit-learn numpy visualkeras
   ```

2. **Run the notebook**:
   Open `main.ipynb` in Jupyter Notebook or Jupyter Lab and execute all cells sequentially.

## 📈 Results

### Model Performance

| Model | Training Accuracy | Test Accuracy | Test Loss |
|-------|------------------|---------------|-----------|
| Simple CNN | 100% | 100% | 0.0020 |
| LeNet | 100% | 100% | 0.0015 |

### Training Features

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Validation Split**: 15%
- **Early Stopping**: Patience of 2 epochs
- **Maximum Epochs**: 20

Both models converged quickly and achieved perfect classification on the test set, indicating that the dataset is well-suited for CNN-based classification.

## 📁 File Structure

```
├── main.ipynb              # Main Jupyter notebook with full implementation
├── lego-train.pickle       # Training dataset (1.0MB)
├── lego-test.pickle        # Test dataset (345KB)
└── README.md              # This file
```

## 🔍 Key Features

- **Data Visualization**: Random sample display with true labels
- **Model Comparison**: Side-by-side evaluation of two architectures
- **Training Monitoring**: Real-time accuracy and loss tracking
- **Confusion Matrix**: Detailed classification performance analysis
- **Model Architecture Visualization**: Visual representation using VisualKeras

## 🎯 Future Improvements

- Experiment with data augmentation techniques
- Test on larger, more diverse LEGO datasets
- Implement transfer learning with pre-trained models
- Add real-time classification from camera input
- Explore ensemble methods for improved robustness

## 📄 License

This project is available for educational and research purposes.

---

**Note**: The exceptionally high accuracy (100%) suggests this dataset may be relatively simple or well-curated. In real-world applications, additional validation on more diverse data would be recommended.