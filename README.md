# KidneyNet-TL
Deep Learning pipeline for Kidney Disease Classification using Transfer Learning on CT scan images.

# Tools & Libraries
TensorFlow, Keras , OpenCV , NumPy, pandas, Scikit-learn, Matplotlib, Seaborn 

# Dataset Reference
This model is trained on a comprehensive kidney CT image dataset with 12,446 labeled images resized to 256x256 pixels. The dataset includes images categorized as Normal, Cyst, Stone, and Tumor. 

https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone

This project presents a Deep Learning pipeline for classifying kidney diseases from CT scan images into four categories: Normal, Cyst, Stone, and Tumor. By using transfer learning, the pipeline utilizes pretrained Convolutional Neural Networks (CNNs) for accurate feature extraction on medical images.

# Features
- Uses ImageNet pretrained CNNs including ResNet50, Xception, and EfficientNetB0 as base models.

- Applies preprocessing techniques such as image resizing, normalization, and label encoding.

- Employs stratified K-Fold cross-validation and randomized hyperparameter tuning for training.

- Constructs classifier heads with global average pooling, batch normalization, dropout, and dense layers.

- Evaluates performance with accuracy metrics, classification reports, and confusion matrices.
