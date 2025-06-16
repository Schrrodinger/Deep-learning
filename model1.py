import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. IMPORT YOUR PREPARED DATA ---
# This line imports the main function from your other file.
# It will run the entire data preparation process and give us the datasets.
try:
    from BinaryDataPreprocessor import run_full_pipeline
except ImportError:
    print("ERROR: Make sure 'train_cnn.py' is in the same folder as 'BinaryDataPreprocessor.py'")
    exit()


# --- 2. DEFINE THE CNN MODEL ARCHITECTURE ---
def build_model(input_shape):
    """Creates and compiles the CNN model."""
    model = Sequential([
        tf.keras.Input(shape=input_shape),

        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Flatten the 2D feature maps into a 1D vector
        Flatten(),

        # Fully Connected Layers for Classification
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary (0 or 1) output
    ])

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    return model

# --- 3. PLOT TRAINING HISTORY ---
def plot_history(history):
    """Plots the training and validation accuracy and loss."""
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

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_history.png')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()


# --- 4. MAIN FUNCTION TO RUN EVERYTHING ---
if __name__ == '__main__':
    # --- Configuration Parameters ---
    NUM_SAMPLES_TO_VISUALIZE = 5
    BATCH_SIZE = 16
    EPOCHS = 25 # You can start with 25 and increase if needed

    # --- Run Data Preprocessing ---
    print("--- STEP 1: Starting Data Preprocessing ---")
    # This calls your other script to prepare the data
    train_ds, val_ds, test_ds, preprocessor, _ = run_full_pipeline(
        num_samples=NUM_SAMPLES_TO_VISUALIZE,
        batch_size=BATCH_SIZE
    )

    # --- Get Image Shape for the Model ---
    # We inspect the dataset to find the image dimensions
    for images, labels in train_ds.take(1):
        input_shape = images.shape[1:]
        break
    
    # --- Build the CNN Model ---
    print("\n--- STEP 2: Building the CNN Model ---")
    model = build_model(input_shape)
    model.summary()

    # --- Train the Model ---
    print("\n--- STEP 3: Starting Model Training ---")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

    # --- Evaluate the Model ---
    print("\n--- STEP 4: Evaluating the Model on Unseen Test Data ---")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # --- Plot Results and Save Model ---
    plot_history(history)
    model.save("binary_cnn_model.h5")
    print("\n--- FINAL STEP: Model saved as 'binary_cnn_model.h5' ---")