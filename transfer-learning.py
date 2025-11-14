import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import itertools
import time
import random
import logging
import os
from sklearn.utils import shuffle
import json


#Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm


from tensorflow.keras import backend as K
import gc

from sklearn.model_selection import StratifiedKFold

data_path  = '/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
labels = ['Cyst','Normal','Stone','Tumor']

X = []
y = []


image_size = 256
for i in labels:
    folderPath = os.path.join(data_path,i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X.append(img)
        y.append(i)

X = np.array(X)
y = np.array(y)

models_dict = {
    "ResNet50": tf.keras.applications.ResNet50,
    "Xception": tf.keras.applications.Xception,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0
}

configs = {
    "Random_Search": {
        "model_name": "ResNet50",
        "lr": 0.0001,
        "batch_size": 16,
        "dropout_rate": 0.4,
        "dense_units": 512,
        "optimizer": "rmsprop",
        "trainable_layers": "last_30"
    }
}

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=1)

print(f"Data shape : {X.shape}")

unique_labels = np.unique(y)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_labels = np.array([label_to_index[i] for i in y])

n_splits = 4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# Store final results per config
final_results = {}

for name, config in configs.items():
    print(f"\n{'='*20}\nRunning config: {name}\n{'='*20}, {config}")

    fold_idx = 1
    fold_accuracies = []
    fold_training_times = []

    for train_index, test_index in skf.split(X, y_labels):
        print(f"\n--- Fold {fold_idx} ---")

        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv = tf.keras.utils.to_categorical(y_labels[train_index], num_classes=len(unique_labels))
        y_test_cv = tf.keras.utils.to_categorical(y_labels[test_index], num_classes=len(unique_labels))

        BaseModelClass = models_dict[config['model_name']]
        base_model = BaseModelClass(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

        if config['trainable_layers'] == "none":
            base_model.trainable = False
        elif config['trainable_layers'].startswith("last_"):
            n = int(config['trainable_layers'].split('_')[1])
            for layer in base_model.layers[:-n]:
                layer.trainable = False
        else:
            base_model.trainable = True

        model = base_model.output
        model = tf.keras.layers.GlobalAveragePooling2D()(model)
        model = tf.keras.layers.Dense(config["dense_units"], activation=None)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Activation('relu')(model)
        model = tf.keras.layers.Dropout(rate=config["dropout_rate"])(model)
        model = tf.keras.layers.Dense(len(unique_labels), activation='softmax')(model)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=model)

        optimizer = {
            "adam": tf.keras.optimizers.Adam(learning_rate=config['lr']),
            "sgd": tf.keras.optimizers.SGD(learning_rate=config['lr']),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=config['lr'])
        }[config['optimizer']]

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        start_time = time.time()
        model.fit(X_train_cv, y_train_cv, validation_split=0.1, epochs=12, verbose=0,
                  batch_size=config['batch_size'], callbacks=[])
        elapsed = time.time() - start_time

        print(f"Training time for {name} fold {fold_idx}: {elapsed:.2f} seconds")

        y_true_test = np.argmax(y_test_cv, axis=1)
        y_pred_test = np.argmax(model.predict(X_test_cv), axis=1)

        acc = accuracy_score(y_true_test, y_pred_test)

        # Plot confusion matrix
        cm = confusion_matrix(y_true_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {name} Fold {fold_idx}")
        plt.show()

        
        fold_accuracies.append(acc)
        fold_training_times.append(elapsed)

        fold_idx += 1

    mean_acc = np.mean(fold_accuracies)
    mean_time = np.mean(fold_training_times)

    final_results[name] = {
        "mean_accuracy": round(mean_acc, 4),
        "mean_training_time": round(mean_time, 2)
    }

# Print final summary
print("\nFinal Summary:")
for name, res in final_results.items():
    print(f"Config: {name} | Mean Accuracy: {res['mean_accuracy']} | Mean Training Time: {res['mean_training_time']} sec")