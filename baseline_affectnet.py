#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = 48
NUM_CLASSES = 8
EPOCHS = 150
BATCH_SIZE = 128
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.01
DATA_DIR = 'preprocessed_data'  # Folder containing X_train.npy, y_train.npy, X_test.npy, y_test.npy

# -------------------------------
# Load Preprocessed Data (Training & Test Sets Only)
# -------------------------------
def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(DATA_DIR)
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# -------------------------------
# Build Model (Same Architecture as FER2013 Baseline)
# -------------------------------
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1"),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(32, (3, 3), activation='relu', padding='same', name="conv2"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), name="maxpool2"),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu', padding='same', name="conv3"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), name="maxpool3"),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu', padding='same', name="conv4"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), name="maxpool4"),
    Dropout(0.2),
    Flatten(),
    Dense(1024, activation='relu', name='fc1'),
    Dropout(DROPOUT_RATE),
    BatchNormalization(),
    Dense(NUM_CLASSES, activation='softmax', name='fcsoftmax')
])

sgd = SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# -------------------------------
# Set Up Callbacks
# -------------------------------
checkpoint = ModelCheckpoint("AffectNet_model_best.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
rlrop = ReduceLROnPlateau(monitor='loss', mode='min', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
callbacks_list = [checkpoint, rlrop]

# -------------------------------
# Train the Model (No Validation Set)
# -------------------------------
history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=callbacks_list,
                    shuffle=True)

# -------------------------------
# Evaluate the Model on Test Set
# -------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print("Test loss:", test_loss, "Test accuracy:", test_acc)

# -------------------------------
# Plot Confusion Matrix on Test Set
# -------------------------------
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_true, y_test_pred_classes, normalize='true')
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
plt.title("Test Set Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -------------------------------
# Save the Final Model
# -------------------------------
model.save("AffectNet_final_model.h5")
print("Model saved as AffectNet_final_model.h5")
