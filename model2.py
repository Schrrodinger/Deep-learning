import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
### Import the VGG16 model and its specific preprocessing function
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

#--------------------------------------------------------------------------
# PART 1: DATA PREPROCESSING CLASS (MODIFIED FOR VGG16)
#--------------------------------------------------------------------------
class VGG16DataPreprocessor:
    def __init__(self, csv_path: str, image_base_path: str, target_size: tuple = (224, 224)):
        self.csv_path = Path(csv_path)
        self.image_base_path = Path(image_base_path)
        ### VGG16 works best with 224x224 images
        self.target_size = target_size
        self.binary_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_and_split_data(self):
        """Load and split the data into train, validation, and test sets."""
        print("Loading and filtering data...")
        full_df = pd.read_csv(self.csv_path)
        
        # Create binary labels based on the 'N' column (Normal vs Abnormal)
        # If N=1, it's Normal (0), otherwise it's Abnormal (1)
        full_df['binary_label'] = 1 - full_df['N']  # Invert N column to get binary label
        
        # Update file paths to use the correct image directory
        full_df['filepath'] = full_df['filepath'].apply(
            lambda x: os.path.join(self.image_base_path, os.path.basename(x))
        )
        
        # Split the data
        train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42, stratify=full_df['binary_label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['binary_label'])
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        print(f"Example filepath: {train_df['filepath'].iloc[0]}")  # Print an example path to verify

    def _process_path(self, file_path: tf.Tensor, label: tf.Tensor):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.target_size)
        ### This is the crucial preprocessing step required for VGG16
        img = preprocess_input(img)
        return img, label

    def create_tf_datasets(self, batch_size: int = 32):
        if self.train_df is None:
            self.load_and_split_data()

        print("Creating TensorFlow datasets...")
        train_ds = tf.data.Dataset.from_tensor_slices((self.train_df['filepath'].values, self.train_df['binary_label'].values))
        val_ds = tf.data.Dataset.from_tensor_slices((self.val_df['filepath'].values, self.val_df['binary_label'].values))
        test_ds = tf.data.Dataset.from_tensor_slices((self.test_df['filepath'].values, self.test_df['binary_label'].values))
        
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(self._process_path, num_parallel_calls=AUTOTUNE).cache().shuffle(len(self.train_df)).batch(batch_size).prefetch(AUTOTUNE)
        val_ds = val_ds.map(self._process_path, num_parallel_calls=AUTOTUNE).batch(batch_size).cache().prefetch(AUTOTUNE)
        test_ds = test_ds.map(self._process_path, num_parallel_calls=AUTOTUNE).batch(batch_size).cache().prefetch(AUTOTUNE)
        
        return train_ds, val_ds, test_ds

#--------------------------------------------------------------------------
# PART 2: VGG16 MODEL BUILDING
#--------------------------------------------------------------------------
def build_vgg16_model(input_shape):
    """Builds a transfer learning model using VGG16."""
    
    # Load the VGG16 base model, pre-trained on ImageNet
    # include_top=False removes its final classification layer
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Freeze the convolutional base. We don't want to re-train these layers.
    base_model.trainable = False

    # Add our own classifier on top of the VGG16 base
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), # Reduces dimensions nicely
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Our final output layer
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    print("VGG16-based transfer learning model built successfully.")
    model.summary()
    return model

def plot_training_history(history):
    # (This function remains the same as before)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('vgg16_training_history.png')
    plt.show()

#--------------------------------------------------------------------------
# PART 3: MAIN EXECUTION BLOCK
#--------------------------------------------------------------------------
def main():
    # Initialize preprocessor
    preprocessor = VGG16DataPreprocessor(
        csv_path="data/full_df.csv",  # Updated path to point to data directory
        image_base_path="data/preprocessed_images",
        target_size=(224, 224)  # VGG16 requires 224x224 input
    )
    train_ds, val_ds, test_ds = preprocessor.create_tf_datasets(batch_size=BATCH_SIZE)

    # --- 2. Build the VGG16 Model ---
    print("\n--- Building the VGG16 Model ---")
    model = build_vgg16_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # --- 3. Train the Model ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # --- 4. Evaluate and Save ---
    print("\n--- Evaluating Model on Test Set ---")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plot_training_history(history)
    model.save("vgg16_model.h5")
    print("\nModel saved as vgg16_model.h5")

if __name__ == '__main__':
    # --- Configuration ---
    # !!! IMPORTANT: UPDATE THESE PATHS TO MATCH YOUR SYSTEM !!!
    CSV_FILE_PATH = "full_df.csv"
    IMAGE_DIRECTORY_PATH = "D:\\Deep_learning\\proj\\data\\preprocessed_images"
    
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20

    main()