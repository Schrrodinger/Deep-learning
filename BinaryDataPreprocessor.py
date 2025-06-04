import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path
import cv2
from typing import Tuple, List, Optional
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

class BinaryDataPreprocessor:
    """
    Binary Data Preprocessing and Loading Pipeline for Medical Image Classification
    """

    def __init__(self, csv_path: str, image_base_path: str, target_size: Tuple[int, int] = (512, 512)):
        self.csv_path = csv_path
        self.image_base_path = image_base_path
        self.target_size = target_size
        self.full_df = None
        self.binary_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_and_create_binary_labels(self, save_path: str = "binary_classification_df.csv") -> pd.DataFrame:
        print("Loading full_df.csv...")
        self.full_df = pd.read_csv(self.csv_path)

        print(f"Original dataset has {len(self.full_df)} samples")
        print("Columns in dataset:", self.full_df.columns.tolist())

        if 'filepath' in self.full_df.columns:
            self.full_df['filepath'] = self.full_df['filepath'].apply(lambda x: os.path.basename(x))
            print("Normalized 'filepath' column to contain only file names")
        elif 'filename' in self.full_df.columns:
            self.full_df['filepath'] = self.full_df['filename']
            print("Using 'filename' column as 'filepath'")
        elif 'image_id' in self.full_df.columns:
            self.full_df['filepath'] = self.full_df['image_id'].apply(lambda x: f"{x}.jpg")
            print("Constructed 'filepath' from 'image_id' assuming .jpg extension")
        else:
            raise ValueError("No suitable column ('filepath', 'filename', or 'image_id') found in the DataFrame!")

        missing_files = []
        for idx, row in self.full_df.iterrows():
            file_path = os.path.join(self.image_base_path, row['filepath'])
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            with open("missing_files_initial.log", "w") as f:
                f.write("Missing files before preprocessing:\n")
                f.write("\n".join(missing_files))
            print(f"Warning: {len(missing_files)} files are missing. Logged to 'missing_files_initial.log'.")
            print("Removing rows with missing files from the DataFrame...")
            self.full_df = self.full_df[self.full_df['filepath'].apply(
                lambda x: os.path.exists(os.path.join(self.image_base_path, x))
            )]
            print(f"Dataset after removing missing files: {len(self.full_df)} samples")

        disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        available_disease_cols = [col for col in disease_columns if col in self.full_df.columns]

        if available_disease_cols:
            print("\nDisease label statistics:")
            for col in available_disease_cols:
                print(f"{col}: {self.full_df[col].sum()} samples")

        if 'N' in self.full_df.columns:
            self.full_df['is_abnormal'] = (self.full_df['N'] == 0).astype(int)
            print("\nCreated binary labels based on 'N' column")
        elif available_disease_cols:
            abnormal_mask = self.full_df[available_disease_cols].sum(axis=1) > 0
            self.full_df['is_abnormal'] = abnormal_mask.astype(int)
            print(f"\nCreated binary labels based on columns: {available_disease_cols}")
        elif 'labels' in self.full_df.columns:
            self.full_df['is_abnormal'] = (self.full_df['labels'] != 'N').astype(int)
            print("\nCreated binary labels based on 'labels' column")
        elif 'target' in self.full_df.columns:
            self.full_df['is_abnormal'] = self.full_df['target']
            print("\nUsing 'target' column as binary label")
        else:
            raise ValueError("No suitable column found to create binary labels!")

        if 'image_quality' in self.full_df.columns:
            initial_count = len(self.full_df)
            self.full_df = self.full_df[self.full_df['image_quality'] != 'Low']
            removed_count = initial_count - len(self.full_df)
            print(f"Removed {removed_count} low quality images")

        if 'image_id' in self.full_df.columns:
            initial_count = len(self.full_df)
            self.full_df = self.full_df.drop_duplicates(subset=['image_id'])
            removed_count = initial_count - len(self.full_df)
            print(f"Removed {removed_count} duplicate images")
        elif 'filepath' in self.full_df.columns:
            initial_count = len(self.full_df)
            self.full_df = self.full_df.drop_duplicates(subset=['filepath'])
            removed_count = initial_count - len(self.full_df)
            print(f"Removed {removed_count} duplicate images based on filepath")

        print(f"\nFinal dataset has {len(self.full_df)} samples")
        normal_count = (self.full_df['is_abnormal'] == 0).sum()
        abnormal_count = (self.full_df['is_abnormal'] == 1).sum()
        print(f"Normal samples: {normal_count} ({normal_count / len(self.full_df) * 100:.1f}%)")
        print(f"Abnormal samples: {abnormal_count} ({abnormal_count / len(self.full_df) * 100:.1f}%)")

        self.binary_df = self.full_df.copy()
        self.binary_df.to_csv(save_path, index=False)
        print(f"\nSaved processed DataFrame to {save_path}")

        return self.binary_df

    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.1,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.binary_df is None:
            raise ValueError("Please run load_and_create_binary_labels() first!")

        print("Splitting dataset...")

        train_val_df, test_df = train_test_split(
            self.binary_df,
            test_size=test_size,
            stratify=self.binary_df['is_abnormal'],
            random_state=random_state
        )

        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['is_abnormal'],
            random_state=random_state
        )

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        print(f"\nDataset split:")
        print(f"Training set: {len(train_df)} samples ({len(train_df) / len(self.binary_df) * 100:.1f}%)")
        print(f"Validation set: {len(val_df)} samples ({len(val_df) / len(self.binary_df) * 100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df) / len(self.binary_df) * 100:.1f}%)")

        for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            normal_pct = (df['is_abnormal'] == 0).mean() * 100
            abnormal_pct = (df['is_abnormal'] == 1).mean() * 100
            print(f"{name} - Normal: {normal_pct:.1f}%, Abnormal: {abnormal_pct:.1f}%")

        return train_df, val_df, test_df

    def organize_image_files(self, output_base_path: str, copy_files: bool = True,
                             image_col: str = 'filepath') -> dict:
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Please run split_dataset() first!")

        if os.path.exists(output_base_path):
            print(f"Clearing existing directory: {output_base_path}")
            shutil.rmtree(output_base_path)

        print("Organizing image files into directories...")

        base_path = Path(output_base_path)
        dirs = {
            'train': {
                'normal': base_path / 'train' / 'normal',
                'abnormal': base_path / 'train' / 'abnormal'
            },
            'val': {
                'normal': base_path / 'val' / 'normal',
                'abnormal': base_path / 'val' / 'abnormal'
            },
            'test': {
                'normal': base_path / 'test' / 'normal',
                'abnormal': base_path / 'test' / 'abnormal'
            }
        }

        for split_dirs in dirs.values():
            for class_dir in split_dirs.values():
                class_dir.mkdir(parents=True, exist_ok=True)

        missing_files = []

        def organize_split(df: pd.DataFrame, split_name: str):
            print(f"Organizing {split_name} files...")
            processed_files = 0
            for idx, row in df.iterrows():
                src_path = None
                if image_col in df.columns:
                    src_path = os.path.join(self.image_base_path, row[image_col])
                elif 'image_id' in df.columns:
                    image_id = row['image_id']
                    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
                    for ext in extensions:
                        potential_path = os.path.join(self.image_base_path, f"{image_id}{ext}")
                        if os.path.exists(potential_path):
                            src_path = potential_path
                            break
                else:
                    print(f"Warning: Cannot determine image file path for row {idx}")
                    missing_files.append(f"Row {idx}: No valid path")
                    continue

                if not src_path or not os.path.exists(src_path):
                    print(f"Warning: File not found - {src_path}")
                    missing_files.append(src_path)
                    continue

                label = row['is_abnormal']
                dst_dir = dirs[split_name]['normal'] if label == 0 else dirs[split_name]['abnormal']

                filename = os.path.basename(src_path)
                dst_path = dst_dir / filename

                counter = 1
                original_dst_path = dst_path
                while dst_path.exists():
                    name, ext = os.path.splitext(original_dst_path.name)
                    dst_path = dst_dir / f"{name}_{counter}{ext}"
                    counter += 1

                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)
                    processed_files += 1
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
                    missing_files.append(f"Error processing {src_path}: {e}")
            print(f"Processed {processed_files} files for {split_name} split")

        organize_split(self.train_df, 'train')
        organize_split(self.val_df, 'val')
        organize_split(self.test_df, 'test')

        if missing_files:
            with open(os.path.join(output_base_path, 'missing_files.log'), 'w') as f:
                f.write("Missing or problematic files:\n")
                f.write("\n".join(missing_files))
            print(f"Logged {len(missing_files)} missing or problematic files to {output_base_path}/missing_files.log")

        print("\nFile organization complete!")
        for split_name, split_dirs in dirs.items():
            normal_count = len(list(split_dirs['normal'].glob('*')))
            abnormal_count = len(list(split_dirs['abnormal'].glob('*')))
            total_count = normal_count + abnormal_count
            print(f"{split_name.capitalize()} set: {total_count} files "
                  f"(Normal: {normal_count}, Abnormal: {abnormal_count})")

        return dirs

    def create_directory_based_datasets(self, organized_dirs: dict, batch_size: int = 32,
                                        validation_split: Optional[float] = None) -> Tuple[
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        print("Creating datasets from organized directories...")

        def load_image(image, label):
            image = tf.cast(image, tf.float32)
            return image, label

        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(Path(organized_dirs['train']['normal']).parent),
            labels='inferred',
            label_mode='binary',
            class_names=['normal', 'abnormal'],
            color_mode='rgb',
            batch_size=None,
            image_size=self.target_size,
            shuffle=True,
            seed=42,
            validation_split=validation_split,
            subset='training' if validation_split else None
        )
        print(f"Training dataset created with {train_ds.cardinality().numpy()} elements before batching")
        train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size)
        print(f"Training dataset after batching: {train_ds.cardinality().numpy()} batches")

        if validation_split:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(Path(organized_dirs['train']['normal']).parent),
                labels='inferred',
                label_mode='binary',
                class_names=['normal', 'abnormal'],
                color_mode='rgb',
                batch_size=None,
                image_size=self.target_size,
                shuffle=True,
                seed=42,
                validation_split=validation_split,
                subset='validation'
            )
        else:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(Path(organized_dirs['val']['normal']).parent),
                labels='inferred',
                label_mode='binary',
                class_names=['normal', 'abnormal'],
                color_mode='rgb',
                batch_size=None,
                image_size=self.target_size,
                shuffle=False,
                seed=42
            )
        print(f"Validation dataset created with {val_ds.cardinality().numpy()} elements before batching")
        val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size)
        print(f"Validation dataset after batching: {val_ds.cardinality().numpy()} batches")

        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(Path(organized_dirs['test']['normal']).parent),
            labels='inferred',
            label_mode='binary',
            class_names=['normal', 'abnormal'],
            color_mode='rgb',
            batch_size=None,
            image_size=self.target_size,
            shuffle=False,
            seed=42
        )
        test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size)

        train_ds = train_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        test_ds = test_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        data_augmentation = self.create_data_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    @tf.function
    def parse_image_and_binary_label(self, image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, self.target_size)
        image.set_shape([*self.target_size, 3])
        image = preprocess_input(image)
        return image, label

    def create_data_augmentation(self) -> tf.keras.Sequential:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1, fill_mode='reflect'),
            tf.keras.layers.RandomZoom(0.1, fill_mode='reflect'),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
        return data_augmentation

    def load_and_prepare_binary_dataset(self, df: pd.DataFrame, batch_size: int = 32,
                                       is_training: bool = False,
                                       image_col: str = 'filepath',
                                       cache: bool = True,
                                       merge_eyes: bool = False) -> tf.data.Dataset:
        if merge_eyes and 'ID' in df.columns:
            grouped = df.groupby('ID').agg({
                image_col: lambda x: list(x),
                'is_abnormal': 'first'
            }).reset_index()
            image_paths = grouped[image_col].tolist()
        else:
            image_paths = df[image_col].apply(lambda x: os.path.join(self.image_base_path, x)).tolist()

        labels = df['is_abnormal'].tolist() if not merge_eyes else grouped['is_abnormal'].tolist()

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(
            lambda x, y: (self.parse_image_and_binary_label(x if not merge_eyes else x[0], y),
                          y) if not merge_eyes else tuple(tf.map_fn(lambda img: self.parse_image_and_binary_label(img, y), x, dtype=tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if cache:
            dataset = dataset.cache()

        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
            data_augmentation = self.create_data_augmentation()
            dataset = dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def visualize_samples(self, dataset: tf.data.Dataset, num_samples: int = 8,
                          batch_size: int = 8, figsize: Tuple[int, int] = (12, 8)) -> None:
        processed_save_dir = "debug_processed_images"
        os.makedirs(processed_save_dir, exist_ok=True)

        num_batches = (num_samples + batch_size - 1) // batch_size
        print(f"Visualizing {num_samples} samples, requiring {num_batches} batch(es)")

        all_images = []
        all_labels = []
        total_images = 0
        for batch in dataset.take(num_batches):
            images, labels = batch
            all_images.append(images.numpy())
            all_labels.append(labels.numpy())
            total_images += len(images)
            print(f"Batch contains {len(images)} images, total collected: {total_images}")
            if total_images >= num_samples:
                break

        if total_images == 0:
            print("Warning: No images collected from the dataset!")
            return

        images_display = np.concatenate(all_images, axis=0)[:num_samples]
        labels = np.concatenate(all_labels, axis=0)[:num_samples]
        print(f"Using {len(images_display)} images for visualization")

        images_display = images_display + [103.939, 116.779, 123.68]
        images_display = images_display[..., ::-1]

        print("Pixel value range before clipping:", images_display.min(), images_display.max())
        images_display = np.clip(images_display, 0, 255) / 255.0
        print("Pixel value range after clipping:", images_display.min(), images_display.max())

        if len(images_display) > 0:
            first_image = images_display[0]
            red_mask = (first_image[:, :, 0] > 200) & (first_image[:, :, 1] < 50) & (first_image[:, :, 2] < 50)
            red_indices = np.where(red_mask)
            if len(red_indices[0]) > 0:
                print("Found potential red line pixels in first image:")
                for idx in range(min(5, len(red_indices[0]))):
                    y, x = red_indices[0][idx], red_indices[1][idx]
                    pixel_value = first_image[y, x, :]
                    print(f"Pixel at ({y}, {x}): {pixel_value}")
            else:
                print("No potential red line pixels found.")

        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows * cols > 1 else [axes]

        for i in range(min(num_samples, len(images_display))):
            ax = axes[i]
            ax.imshow(images_display[i])
            label_text = "Abnormal" if labels[i] == 1 else "Normal"
            ax.set_title(f"Label: {label_text}")
            ax.axis('off')
            save_path = os.path.join(processed_save_dir, f"processed_image_{i}.jpg")
            plt.imsave(save_path, images_display[i])
            print(f"Saved processed image to {save_path}")

        for i in range(len(images_display), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_raw_images(self, num_samples: int = 8, figsize: Tuple[int, int] = (12, 16)) -> None:
        if self.val_df is None:
            raise ValueError("Please run split_dataset() first!")

        print("Saving raw images from validation set...")

        raw_save_dir = "raw_validation_images"
        os.makedirs(raw_save_dir, exist_ok=True)

        sample_df = self.val_df.sample(n=min(num_samples, len(self.val_df)), random_state=42)
        print(f"Sampled {len(sample_df)} rows from validation set")
        image_paths = sample_df['filepath'].apply(lambda x: os.path.join(self.image_base_path, x)).tolist()
        labels = sample_df['is_abnormal'].tolist()

        print("Image paths to be processed:")
        for path in image_paths:
            print(path)

        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows * cols > 1 else [axes]

        processed_images = 0
        for i, (image_path, label) in enumerate(zip(image_paths, labels)):
            if i >= num_samples:
                break
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                axes[i].axis('off')
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_display = image_rgb / 255.0
            save_path = os.path.join(raw_save_dir, f"raw_image_{processed_images}.jpg")
            plt.imsave(save_path, image_display)
            print(f"Saved raw image to: {save_path}")
            ax = axes[i]
            ax.imshow(image_display)
            label_text = "Abnormal" if label == 1 else "Normal"
            ax.set_title(f"Label: {label_text}")
            ax.axis('off')
            processed_images += 1

        for i in range(processed_images, len(axes)):
            axes[i].axis('off')

        print(f"Processed and saved {processed_images} raw images")
        plt.tight_layout()
        plt.show()

    def get_dataset_info(self, dataset: tf.data.Dataset) -> dict:
        dataset_take = dataset.take(1)
        batch = next(iter(dataset_take))
        images, labels = batch

        info = {
            'batch_size': len(images),
            'image_shape': images.shape[1:],
            'image_dtype': images.dtype,
            'label_dtype': labels.dtype,
            'sample_images': images.numpy()[:10],
            'sample_labels': labels.numpy()[:10],
        }

        return info

def main(num_samples: int = 8):
    image_base_path = "D:\Deep_learning\proj\data\preprocessed_images"
    csv_path = "data/full_df.csv"

    if not os.path.exists(image_base_path):
        raise FileNotFoundError(f"Image base path does not exist: {image_base_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    preprocessor = BinaryDataPreprocessor(
        csv_path=csv_path,
        image_base_path=image_base_path,
        target_size=(512, 512)
    )

    binary_df = preprocessor.load_and_create_binary_labels()

    train_df, val_df, test_df = preprocessor.split_dataset(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    organized_dirs = preprocessor.organize_image_files(
        output_base_path="organized_dataset",
        copy_files=True,
        image_col='filepath'
    )

    batch_size = max(num_samples, 8)
    train_ds, val_ds, test_ds = preprocessor.create_directory_based_datasets(
        organized_dirs=organized_dirs,
        batch_size=batch_size
    )

    print("\nDataset Information:")
    train_info = preprocessor.get_dataset_info(train_ds)
    print("Training dataset info:", train_info)

    print("\nVisualizing raw validation images (directly from disk)...")
    preprocessor.visualize_raw_images(num_samples=num_samples)

    print("\nVisualizing processed training samples (as per requirement)...")
    preprocessor.visualize_samples(train_ds, num_samples=num_samples, batch_size=batch_size)

    print("\nPipeline setup complete!")
    print(f"Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")

    return train_ds, val_ds, test_ds, preprocessor, organized_dirs

def organize_files_only():
    preprocessor = BinaryDataPreprocessor(
        csv_path="data/full_df.csv",
        image_base_path="D:\Deep_learning\proj\data\preprocessed_images",
        target_size=(512, 512)
    )

    binary_df = preprocessor.load_and_create_binary_labels()
    train_df, val_df, test_df = preprocessor.split_dataset()
    organized_dirs = preprocessor.organize_image_files(
        output_base_path="organized_dataset",
        copy_files=True
    )

    print("File organization complete!")
    return organized_dirs

if __name__ == "__main__":
    train_ds, val_ds, test_ds, preprocessor, organized_dirs = main(num_samples=8)