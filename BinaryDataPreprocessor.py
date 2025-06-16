import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path
from typing import Tuple, List, Dict
import cv2

class BinaryDataPreprocessor:
    """
    Handles the full pipeline for binary medical image classification preprocessing.
    """
    def __init__(self, csv_path: str, image_base_path: str, target_size: tuple = (512, 512)):
        self.csv_path = csv_path
        self.image_base_path = image_base_path
        self.target_size = target_size
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        print(f"Preprocessor initialized. Looking for CSV at: {self.csv_path}")
        print(f"Image base path set to: {self.image_base_path}")

    def load_and_create_binary_labels(self):
        """Load the CSV file and create binary labels."""
        self.df = pd.read_csv(self.csv_path)
        
        # Update file paths to use the correct image directory
        self.df['filepath'] = self.df['filepath'].apply(
            lambda x: os.path.join(self.image_base_path, os.path.basename(x))
        )
        
        # Create binary labels (1 for any disease, 0 for normal)
        self.df['binary_label'] = 0
        # If any of the disease columns (N,D,G,C,A,H,M,O) is 1, set binary_label to 1
        disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        self.df.loc[self.df[disease_columns].sum(axis=1) > 0, 'binary_label'] = 1
        
        print(f"Total samples: {len(self.df)}")
        print(f"Normal samples: {len(self.df[self.df['binary_label'] == 0])}")
        print(f"Disease samples: {len(self.df[self.df['binary_label'] == 1])}")
        print(f"Example filepath: {self.df['filepath'].iloc[0]}")  # Print an example path to verify
        
    def split_dataset(self, val_size=0.15, test_size=0.15, random_state=42):
        """Split the dataset into train, validation, and test sets."""
        # First split into train and temp
        train_df, temp_df = train_test_split(
            self.df, 
            test_size=val_size + test_size,
            random_state=random_state,
            stratify=self.df['binary_label']
        )
        
        # Then split temp into validation and test
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1-val_ratio,
            random_state=random_state,
            stratify=temp_df['binary_label']
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        
    def create_tf_datasets(self, batch_size=32):
        """Create TensorFlow datasets for training, validation, and testing."""
        def load_and_preprocess_image(file_path, label):
            # Read the image
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.target_size)
            img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
            return img, label
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices(
            (self.train_df['filepath'].values, self.train_df['binary_label'].values)
        ).map(load_and_preprocess_image).batch(batch_size)
        
        val_ds = tf.data.Dataset.from_tensor_slices(
            (self.val_df['filepath'].values, self.val_df['binary_label'].values)
        ).map(load_and_preprocess_image).batch(batch_size)
        
        test_ds = tf.data.Dataset.from_tensor_slices(
            (self.test_df['filepath'].values, self.test_df['binary_label'].values)
        ).map(load_and_preprocess_image).batch(batch_size)
        
        return train_ds, val_ds, test_ds

def run_full_pipeline(num_samples=None, batch_size=32):
    """
    Run the complete data preprocessing pipeline.
    
    Args:
        num_samples (int, optional): Number of samples to process. If None, process all samples.
        batch_size (int): Batch size for the TensorFlow datasets.
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, preprocessor, df)
    """
    # Initialize the preprocessor
    preprocessor = BinaryDataPreprocessor(
        csv_path='data/full_df.csv',
        image_base_path='data/preprocessed_images'
    )
    
    # Load and create binary labels
    preprocessor.load_and_create_binary_labels()
    
    # If num_samples is specified, limit the dataset
    if num_samples is not None:
        preprocessor.df = preprocessor.df.sample(n=num_samples, random_state=42)
    
    # Split the dataset
    preprocessor.split_dataset()
    
    # Create TensorFlow datasets
    train_ds, val_ds, test_ds = preprocessor.create_tf_datasets(batch_size=batch_size)
    
    return train_ds, val_ds, test_ds, preprocessor, preprocessor.df

if __name__ == "__main__":
    # Example usage
    train_ds, val_ds, test_ds, preprocessor, df = run_full_pipeline()