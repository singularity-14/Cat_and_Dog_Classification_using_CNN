# 🐱🐶 Cat and Dog Classifier using CNN

> A custom Convolutional Neural Network trained from scratch on 25,000 images to classify cats and dogs — achieving **75.26% validation accuracy** in just 4 epochs on a Kaggle GPU.

---

## 📌 Project Overview

This project builds a **binary image classification pipeline** from scratch using a custom CNN architecture. The model is trained on the classic Dogs vs. Cats dataset (Kaggle), demonstrating hands-on proficiency with computer vision, deep learning architecture design, and model evaluation techniques.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏗️ Custom CNN | Built from scratch with Conv2D, BatchNorm, MaxPooling, and Dropout |
| 📊 20,000 training images | Balanced dataset of cats and dogs at 256×256 resolution |
| 📉 Training visualization | Loss and accuracy curves plotted with Seaborn/Matplotlib |
| 🔮 Inference pipeline | Real-time prediction from image input using OpenCV |
| 💾 Model export | Saved as `.h5` for reuse and deployment |
| ⚡ GPU-accelerated | Trained on Kaggle's NVIDIA Tesla T4 GPU |

---

## 🚀 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV (`cv2`) |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Compute | Kaggle (NVIDIA Tesla T4 GPU) |

---

## 🏗️ Model Architecture

```
Input: (256, 256, 3) RGB Image
        ↓
Conv2D(130, 3×3, ReLU) → BatchNormalization → MaxPooling(2×2)
        ↓
Conv2D(60, 3×3, ReLU)  → BatchNormalization → MaxPooling(2×2)
        ↓
Conv2D(32, 3×3, ReLU)  → BatchNormalization → MaxPooling(2×2)
        ↓
Flatten → Dense(120, ReLU) → Dropout(0.1)
        ↓
Dense(32, ReLU) → Dropout(0.1)
        ↓
Dense(1, Sigmoid) → Binary Output (Cat / Dog)
```

**Total trainable parameters: ~3.55M**

---

## 📊 Training Results

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Val Loss |
|-------|---------------|---------------------|------------|----------|
| 1 | 59.89% | 55.36% | 0.7507 | 0.7120 |
| 2 | 66.87% | 68.22% | 0.6106 | 0.5776 |
| 3 | 72.89% | 76.62% | 0.5389 | 0.4973 |
| 4 | 78.36% | **75.26%** | 0.4623 | 0.5424 |

> Best validation accuracy of **76.62%** achieved at epoch 3, trained in just ~9 minutes on GPU.

---

## 📂 Dataset

| Split | Size |
|-------|------|
| Training | 20,000 images |
| Test | 5,000 images |
| Image size | 256 × 256 pixels |
| Classes | Cat (0), Dog (1) |

Source: [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/cat-dog-classifier-cnn.git
cd cat-dog-classifier-cnn
```

### 2. Install dependencies
```bash
pip install tensorflow opencv-python pandas numpy matplotlib seaborn
```

### 3. Run the notebook

Open `cat_and_dog_classifier_cnn.ipynb` in **Kaggle** or Jupyter:
- Enable GPU accelerator in Kaggle settings
- Update dataset paths if running locally
- Run all cells sequentially

### 4. Test with a custom image
```python
import cv2
import numpy as np
from keras.models import load_model

model = load_model('Cat_and_Dog_classifier.h5')

test_img = cv2.imread('your_image.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3)) / 255.0

prediction = model.predict(test_input)
print("Cat" if prediction[0][0] < 0.5 else "Dog")
```

---

## 📂 Project Structure

```
cat-dog-classifier-cnn/
│
├── cat_and_dog_classifier_cnn.ipynb   # Full pipeline: preprocessing, model, training, evaluation
├── Cat_and_Dog_classifier.h5          # Saved trained model (auto-generated after training)
└── README.md                          # Project documentation
```

---

## 💡 Key Learnings & Takeaways

- Designed a **custom CNN architecture** with progressive filter reduction (130 → 60 → 32) for hierarchical feature extraction
- Applied **Batch Normalization** after each convolutional block to stabilize and accelerate training
- Used **Dropout** in dense layers to reduce overfitting on the training set
- Implemented a **pixel normalization function** using `tf.cast` mapped over the dataset pipeline for efficient preprocessing
- Built an **end-to-end inference pipeline** using OpenCV for real-time image prediction

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*A foundational computer vision project exploring CNN architecture design, image preprocessing pipelines, and binary classification.*
