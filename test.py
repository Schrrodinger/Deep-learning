from BinaryDataPreprocessor import BinaryDataPreprocessor
from src.model import build_cnn_model, plot_training_history

if __name__ == '__main__':
    # --- Configuration ---
    # !!! IMPORTANT: UPDATE THESE PATHS TO MATCH YOUR SYSTEM !!!
    CSV_FILE_PATH = "data/full_df.csv"
    IMAGE_DIRECTORY_PATH = "data/preprocessed_images"

    # Model and Training Parameters
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    BATCH_SIZE = 16  # Adjusted for potentially large images
    EPOCHS = 20

    # --- 1. Data Preprocessing ---
    print("--- Starting Data Preprocessing ---")
    preprocessor = BinaryDataPreprocessor(
        csv_path=CSV_FILE_PATH,
        image_base_path=IMAGE_DIRECTORY_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    preprocessor.load_and_create_binary_labels()
    preprocessor.split_dataset()
    train_ds, val_ds, test_ds = preprocessor.create_tf_datasets(batch_size=BATCH_SIZE)

    # --- 2. Build the Model ---
    print("\n--- Building the CNN Model ---")
    model = build_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # --- 3. Train the Model ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # --- 4. Evaluate the Model ---
    print("\n--- Evaluating Model on Test Set ---")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # --- 5. Visualize and Save Results ---
    print("\n--- Plotting Training History ---")
    plot_training_history(history)

    print("\n--- Saving the Final Model ---")
    model.save("binary_cnn_model.h5")
    print("\nModel saved as binary_cnn_model.h5")
    print("Script finished successfully!")