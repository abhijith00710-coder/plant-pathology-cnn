# 🌿 Plant Pathology Prediction using CNN

### CNN Classification of Apple Leaf Diseases with and without Data Augmentation

This project implements a Convolutional Neural Network (CNN) for classifying apple leaf images into four categories:

* 🌱 Healthy
* 🍂 Rust
* 🍁 Scab
* 🦠 Multiple Diseases

The project investigates how **data augmentation affects CNN performance** by comparing the same custom CNN architecture trained with and without augmentation.

A third approach using **EfficientNetB0 transfer learning** was also prepared as the next stage of the project, but its training and evaluation have not yet been completed.

---

## 📌 Project Objective

The main objective is to develop an image classification pipeline for apple leaf disease identification and investigate whether data augmentation improves the generalisation of a CNN trained on a relatively small dataset.

The project compares:

1. **Custom CNN without augmentation**
2. **Custom CNN with augmentation**
3. **EfficientNetB0 transfer learning — training pending**

The first two models use the same CNN architecture and training configuration, allowing the effect of data augmentation to be examined separately.

---

## 📊 Dataset

The project uses the **Plant Pathology 2020 (FGVC7)** dataset.

The original dataset contains **1,821 labelled apple leaf images** across four classes:

| Class             | Images | Share |
| ----------------- | -----: | ----: |
| Rust              |    622 | 34.2% |
| Scab              |    592 | 32.5% |
| Healthy           |    516 | 28.3% |
| Multiple Diseases |     91 |  5.0% |

The dataset is noticeably imbalanced, with the `multiple_diseases` class representing approximately 5% of the original images.

An **80/20 stratified train-test split** was used with `random_state=42`.

---

## 🔄 Data Preprocessing

Images were loaded using OpenCV and resized according to the model being used:

* **100 × 100** for the custom CNN models
* **224 × 224** for the EfficientNetB0 pipeline

The active preprocessing pipeline uses the Keras EfficientNet-specific `preprocess_input` function.

> Note: The current implementation loads images using OpenCV's default BGR channel ordering. This should be corrected to RGB before completing the EfficientNetB0 training stage.

---

## 🔁 Data Augmentation

Five augmentation techniques were implemented manually using OpenCV:

| Technique            | Purpose                           |
| -------------------- | --------------------------------- |
| Rotation ±15°        | Simulates changes in camera angle |
| Horizontal Flip      | Creates left-right variations     |
| Zoom 0.8×–1.2×       | Simulates different distances     |
| Shift ±10 px         | Simulates imperfect framing       |
| Brightness 0.7×–1.3× | Simulates lighting variation      |

Applying the five transformations to the 1,821 original images generated **9,105 augmented images**, producing a total of **10,926 images**.

---

# 🧠 Model 1 — Custom CNN without Augmentation

The first model is a CNN built from scratch using the Keras Sequential API.

### Architecture

```text
Input
  ↓
Conv2D (32)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
MaxPooling
  ↓
Conv2D (64)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
MaxPooling
  ↓
Conv2D (128)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
MaxPooling
  ↓
Flatten
  ↓
Dense (128)
  ↓
Dropout (0.2)
  ↓
Dense (4, Softmax)
```

The model contains approximately **2.45 million parameters**.

### Training configuration

* Optimizer: Adam
* Loss: Categorical Cross-Entropy
* Batch size: 32
* Maximum epochs: 20
* EarlyStopping patience: 8
* Best weights restored using validation loss

### Result

| Metric             |      Score |
| ------------------ | ---------: |
| Test Accuracy      | **78.36%** |
| Weighted Precision | **80.39%** |
| Weighted Recall    | **78.36%** |
| Weighted F1-Score  | **76.97%** |

The training curves showed a substantial train-validation gap, indicating overfitting.

---

# 🚀 Model 2 — Custom CNN with Augmentation

Model 2 uses the **exact same architecture and training configuration** as Model 1.

The only major change is the training data: the augmented dataset was used.

This makes the comparison useful because the model architecture and parameter count remain unchanged.

### Result

| Metric             |      Score |
| ------------------ | ---------: |
| Test Accuracy      | **89.80%** |
| Weighted Precision | **88.23%** |
| Weighted Recall    | **89.80%** |
| Weighted F1-Score  | **88.03%** |

---

# 📈 Model Comparison

| Model                   |   Accuracy |  Precision |     Recall |   F1-Score |
| ----------------------- | ---------: | ---------: | ---------: | ---------: |
| CNN — No Augmentation   |     78.36% |     80.39% |     78.36% |     76.97% |
| CNN — With Augmentation | **89.80%** | **88.23%** | **89.80%** | **88.03%** |
| EfficientNetB0          |    Pending |    Pending |    Pending |    Pending |

The augmented model achieved an approximately **11.4 percentage-point increase in test accuracy** and an approximately **11.1 percentage-point increase in F1-score** compared with the non-augmented model.

Both custom CNN models have the same **2,454,084 parameters**, so the observed difference is associated with the change in training data rather than an increase in model capacity.

---

# 🔬 Model 3 — EfficientNetB0 Transfer Learning

The third stage uses **EfficientNetB0 with ImageNet-pretrained weights**.

The base network is frozen initially, with a custom classification head added on top.

### Classification head

```text
EfficientNetB0
      ↓
Global Average Pooling
      ↓
Batch Normalization
      ↓
Dense (256, ReLU)
      ↓
Dropout (0.4)
      ↓
Dense (128, ReLU)
      ↓
Dropout (0.3)
      ↓
Dense (4, Softmax)
```

### Current status

**Architecture prepared — training pending.**

No accuracy or evaluation metrics are reported for this model yet.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* CNN
* Data Augmentation
* Transfer Learning
* EfficientNetB0
* Google Colab

---

# 📁 Project Structure

```text
plant-pathology-cnn/
│
├── README.md
├── notebooks/
│   └── CNN_plant_pathology.ipynb
│
├── report/
│   └── Abhijith_AB_CNN_Project_Report.pdf
│
├── results/
│   ├── model1_no_augmentation/
│   ├── model2_with_augmentation/
│   ├── class_distribution.png
│   └── model_comparison.png
│
├── images/
├── requirements.txt
└── .gitignore
```

---

# 📚 Key Learning Outcomes

Through this project, I worked on:

* Building a CNN from scratch
* Image preprocessing using OpenCV
* Handling class imbalance
* Implementing image augmentation
* Training and evaluating CNN models
* Monitoring overfitting using training/validation curves
* Using EarlyStopping
* Comparing models using accuracy, precision, recall and F1-score
* Preparing a transfer-learning pipeline using EfficientNetB0

---

# 🔭 Future Work

The next stages of the project are:

* Complete EfficientNetB0 training
* Correct BGR → RGB conversion before transfer learning
* Compare all three approaches
* Generate confusion matrices
* Perform misclassification analysis
* Investigate fine-tuning of the pretrained EfficientNetB0 layers

---

# 📖 References

* Plant Pathology 2020 — FGVC7 dataset
* Thapa et al. — *The Plant Pathology 2020 challenge dataset to classify foliar disease of apples*
* TensorFlow / Keras documentation
* Tan & Le — *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*

---

## 👨‍💻 Author

**Abhijith A B**

BSc Mathematics | Postgraduate in Statistics | Data Science / AI & ML

📍 Trivandrum, Kerala, India
