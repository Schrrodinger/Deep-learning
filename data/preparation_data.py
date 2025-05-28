import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import torch
import torchvision.transforms as transforms
import shutil
import json
import warnings

warnings.filterwarnings('ignore')


# Configuration class
class Config:
    IMAGE_DIR = "D:\Deep learning\project\data\preprocessed_images"
    CSV_PATH = "full_df.csv"
    OUTPUT_DIR = "data/processed"
    TARGET_SIZE = (512, 512)
    LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    SPLIT_RATIO = "70:15:15"  # Options: "70:15:15" or "80:10:10"
    RANDOM_STATE = 42
    IMAGE_ID_COLUMN = "filename"  # Adjust to match your CSV
    COPY_IMAGES = True


class DataPreparationPipeline:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.class_weights = None
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def load_and_inspect_dataset(self):
        """Load and inspect the CSV dataset"""
        print("=" * 50)
        print("STEP 1: DATASET LOADING AND INSPECTION")
        print("=" * 50)

        try:
            self.df = pd.read_csv(self.config.CSV_PATH)
            print(f"✓ Dataset loaded: {len(self.df)} records, shape: {self.df.shape}")
            print("\n--- Dataset Info ---")
            print(self.df.info())
            print("\n--- First 5 Rows ---")
            print(self.df.head())

            missing_values = self.df.isnull().sum()
            print("\n--- Missing Values ---")
            if missing_values.sum() > 0:
                print(missing_values[missing_values > 0])
            else:
                print("✓ No missing values")

            missing_labels = [col for col in self.config.LABEL_COLUMNS if col not in self.df.columns]
            if missing_labels:
                raise ValueError(f"Missing label columns: {missing_labels}")
            print("✓ All label columns present")

            possible_id_columns = ['filename', 'image_id', 'id', 'image_name', 'img_id']
            if self.config.IMAGE_ID_COLUMN not in self.df.columns:
                found_column = None
                for col in possible_id_columns:
                    if col in self.df.columns:
                        found_column = col
                        print(f"⚠️ Specified IMAGE_ID_COLUMN '{self.config.IMAGE_ID_COLUMN}' not found")
                        print(f"   Using detected column '{found_column}' instead")
                        self.config.IMAGE_ID_COLUMN = found_column
                        break
                if not found_column:
                    raise ValueError(
                        f"Image ID column '{self.config.IMAGE_ID_COLUMN}' not found in CSV. "
                        f"Available columns: {list(self.df.columns)}. "
                        f"Expected one of {possible_id_columns} or set IMAGE_ID_COLUMN in Config."
                    )
            print(f"✓ Image ID column '{self.config.IMAGE_ID_COLUMN}' found")

        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            raise

        return self.df

    def analyze_label_distribution(self):
        """Analyze label distribution and class imbalance"""
        print("\n" + "=" * 50)
        print("STEP 2: LABEL DISTRIBUTION ANALYSIS")
        print("=" * 50)

        label_counts = {col: self.df[col].sum() for col in self.config.LABEL_COLUMNS}
        print("--- Individual Label Counts ---")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.df)) * 100
            print(f"{label}: {count:,} samples ({percentage:.2f}%)")

        self.df['label_count'] = self.df[self.config.LABEL_COLUMNS].sum(axis=1)
        print("\n--- Multi-label Statistics ---")
        print(f"Samples with no labels: {(self.df['label_count'] == 0).sum()}")
        print(f"Samples with 1 label: {(self.df['label_count'] == 1).sum()}")
        print(f"Samples with 2+ labels: {(self.df['label_count'] >= 2).sum()}")
        print(f"Average labels per sample: {self.df['label_count'].mean():.2f}")

        self._visualize_label_distribution(label_counts)

        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print("\n--- Class Imbalance Analysis ---")
        print(f"Most common class: {max_count:,} samples")
        print(f"Least common class: {min_count:,} samples")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

        threshold = max_count * 0.1
        imbalanced_classes = [label for label, count in label_counts.items() if count < threshold]
        if imbalanced_classes:
            print(f"⚠️ Severely imbalanced classes (< 10% of max): {imbalanced_classes}")
        else:
            print("✓ No severely imbalanced classes")

        return label_counts

    def _visualize_label_distribution(self, label_counts):
        """Visualize label distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        axes[0, 0].bar(labels, counts, color='skyblue')
        axes[0, 0].set_title('Individual Label Distribution')
        axes[0, 0].set_xlabel('Labels')
        axes[0, 0].set_ylabel('Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)

        label_count_dist = self.df['label_count'].value_counts().sort_index()
        axes[0, 1].bar(label_count_dist.index, label_count_dist.values, color='lightcoral')
        axes[0, 1].set_title('Labels per Sample')
        axes[0, 1].set_xlabel('Number of Labels')
        axes[0, 1].set_ylabel('Samples')

        correlation_matrix = self.df[self.config.LABEL_COLUMNS].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Label Correlation')

        imbalance_ratios = [count / max(counts) for count in counts]
        axes[1, 1].bar(labels, imbalance_ratios, color='orange')
        axes[1, 1].set_title('Class Imbalance Ratios')
        axes[1, 1].set_xlabel('Labels')
        axes[1, 1].set_ylabel('Ratio to Max')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0.1, color='red', linestyle='--', label='10% threshold')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{self.config.OUTPUT_DIR}/label_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def inspect_images(self, sample_size=100):
        """Inspect image files"""
        print("\n" + "=" * 50)
        print("STEP 3: IMAGE INSPECTION")
        print("=" * 50)

        if not os.path.exists(self.config.IMAGE_DIR):
            print(f"❌ Image directory not found: {self.config.IMAGE_DIR}")
            return

        image_files = [f for f in os.listdir(self.config.IMAGE_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        print(f"✓ Found {len(image_files)} image files")

        if not image_files:
            print("❌ No image files found")
            return

        sample_files = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)
        sizes, channels, file_sizes, corrupted = [], [], [], []

        print(f"📊 Inspecting {len(sample_files)} images...")
        for i, img_name in enumerate(sample_files):
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(sample_files)} images...")

            img_path = os.path.join(self.config.IMAGE_DIR, img_name)
            file_sizes.append(os.path.getsize(img_path) / 1024)

            img = cv2.imread(img_path)
            if img is None:
                corrupted.append(img_name)
                continue

            h, w, c = img.shape
            sizes.append((w, h))
            channels.append(c)

        print("\n--- Image Inspection Results ---")
        print(f"✓ Loaded: {len(sizes)} images")
        print(f"❌ Corrupted: {len(corrupted)} images")
        if corrupted:
            print(f"Corrupted files: {corrupted[:5]}{'...' if len(corrupted) > 5 else ''}")

        if sizes:
            unique_sizes = Counter(sizes)
            print(f"\n📏 Image sizes: {len(unique_sizes)} unique")
            print("Most common sizes:")
            for size, count in unique_sizes.most_common(3):
                print(f"   {size[0]}x{size[1]}: {count} images")

            print(f"\n🎨 Channels: {dict(Counter(channels))}")
            print(
                f"\n💾 File sizes (KB): Avg {np.mean(file_sizes):.1f}, Range {np.min(file_sizes):.1f}-{np.max(file_sizes):.1f}")

            self._visualize_image_properties(sizes, file_sizes)

        return {'total_images': len(image_files), 'corrupted': len(corrupted)}

    def check_all_image_sizes(image_dir):
        """Check the sizes of all images in the specified directory."""
        print(f"\nChecking image sizes in {image_dir}...")

        # Collect all image files
        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        print(f"✓ Found {len(image_files)} image files")

        if not image_files:
            print("❌ No image files found")
            return

        # Inspect all images
        sizes = []
        corrupted = []

        print(f"📊 Inspecting {len(image_files)} images...")
        for i, img_name in enumerate(image_files):
            if (i + 1) % 500 == 0 or i + 1 == len(image_files):
                print(f"   Processed {i + 1}/{len(image_files)} images...")

            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                corrupted.append(img_name)
                continue

            h, w, _ = img.shape
            sizes.append((w, h))

        # Report results
        print("\n--- Image Size Results ---")
        print(f"✓ Loaded: {len(sizes)} images")
        print(f"❌ Corrupted: {len(corrupted)} images")
        if corrupted:
            print(f"Corrupted files: {corrupted[:5]}{'...' if len(corrupted) > 5 else ''}")

        if sizes:
            unique_sizes = Counter(sizes)
            print(f"\n📏 Image sizes: {len(unique_sizes)} unique")
            print("All sizes:")
            for size, count in unique_sizes.most_common():
                print(f"   {size[0]}x{size[1]}: {count} images")

        return {'total_images': len(image_files), 'corrupted': len(corrupted), 'sizes': sizes}

    # Example usage
    image_dir = "D:\Deep learning\project\data\preprocessed_images"
    result = check_all_image_sizes(image_dir)

    def _visualize_image_properties(self, sizes, file_sizes):
        """Visualize image properties"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        if sizes:
            widths, heights = zip(*sizes)
            axes[0].scatter(widths, heights, alpha=0.6, s=20)
            axes[0].set_xlabel('Width (px)')
            axes[0].set_ylabel('Height (px)')
            axes[0].set_title('Image Dimensions')
            axes[0].grid(True, alpha=0.3)

            aspect_ratios = [w / h for w, h in sizes]
            axes[1].hist(aspect_ratios, bins=20, alpha=0.7, color='orange')
            axes[1].set_xlabel('Aspect Ratio (W/H)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Aspect Ratios')
            axes[1].axvline(x=1.0, color='red', linestyle='--', label='1:1')
            axes[1].legend()

        axes[2].hist(file_sizes, bins=20, alpha=0.7, color='green')
        axes[2].set_xlabel('File Size (KB)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('File Sizes')

        plt.tight_layout()
        plt.savefig(f"{self.config.OUTPUT_DIR}/image_properties.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_data_splits(self):
        """Split dataset into train/val/test with multi-label stratification"""
        print("\n" + "=" * 50)
        print("STEP 4: DATA SPLITTING")
        print("=" * 50)

        if self.config.SPLIT_RATIO == "70:15:15":
            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        elif self.config.SPLIT_RATIO == "80:10:10":
            train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        else:
            raise ValueError("SPLIT_RATIO must be '70:15:15' or '80:10:10'")

        # Create stratification column
        self.df['label_count'] = self.df[self.config.LABEL_COLUMNS].sum(axis=1)
        self.df['stratify_label'] = self.df[self.config.LABEL_COLUMNS].apply(
            lambda x: ''.join(x.astype(str)), axis=1) + "_" + self.df['label_count'].astype(str)

        # Identify rare combinations (fewer than 2 samples)
        strat_counts = self.df['stratify_label'].value_counts()
        rare_combinations = strat_counts[strat_counts < 2].index.tolist()

        if rare_combinations:
            print(f"⚠️ Found {len(rare_combinations)} rare label combinations (fewer than 2 samples):")
            print(f"   Examples: {rare_combinations[:3]}{'...' if len(rare_combinations) > 3 else ''}")
            print(f"   These will be grouped into an 'other' category for stratification")

            # Group rare combinations into an 'other' category
            self.df['stratify_label'] = self.df['stratify_label'].apply(
                lambda x: 'other' if x in rare_combinations else x
            )

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_ratio,
            random_state=self.config.RANDOM_STATE,
            stratify=self.df['stratify_label']
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.config.RANDOM_STATE,
            stratify=train_val_df['stratify_label']
        )

        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

        print(f"✓ Data split completed ({self.config.SPLIT_RATIO}):")
        print(f"   Training: {len(train_df):,} ({len(train_df) / len(self.df) * 100:.1f}%)")
        print(f"   Validation: {len(val_df):,} ({len(val_df) / len(self.df) * 100:.1f}%)")
        print(f"   Test: {len(test_df):,} ({len(test_df) / len(self.df) * 100:.1f}%)")

        self._verify_multilabel_representation()

        self.train_df.to_csv(f"{self.config.OUTPUT_DIR}/train_split.csv", index=False)
        self.val_df.to_csv(f"{self.config.OUTPUT_DIR}/val_split.csv", index=False)
        self.test_df.to_csv(f"{self.config.OUTPUT_DIR}/test_split.csv", index=False)
        print(f"✓ Split files saved to {self.config.OUTPUT_DIR}")

        if self.config.COPY_IMAGES:
            self._split_image_files()

        return self.train_df, self.val_df, self.test_df

    def _verify_multilabel_representation(self):
        """Verify multi-label representation in test set"""
        print("\n--- Multi-label Representation in Test Set ---")
        overall_multi_label = (self.df['label_count'] >= 2).sum() / len(self.df) * 100
        test_multi_label = (self.test_df['label_count'] >= 2).sum() / len(self.test_df) * 100

        print(f"Overall: {overall_multi_label:.1f}% samples with 2+ labels")
        print(f"Test set: {test_multi_label:.1f}% samples with 2+ labels")

        if abs(overall_multi_label - test_multi_label) > 5.0:
            print("⚠️ Warning: Test set multi-label representation deviates >5%")
        else:
            print("✓ Test set represents multi-label cases well")

        print("\n--- Test Set Label Distribution ---")
        for col in self.config.LABEL_COLUMNS:
            count = self.test_df[col].sum()
            percentage = (count / len(self.test_df)) * 100
            print(f"   {col}: {count:,} ({percentage:.1f}%)")

    def _split_image_files(self):
        """Copy images to train/val/test directories"""
        print("\n" + "=" * 50)
        print("STEP 4.1: SPLITTING IMAGE FILES")
        print("=" * 50)

        split_dirs = {
            'train': os.path.join(self.config.OUTPUT_DIR, 'images/train'),
            'val': os.path.join(self.config.OUTPUT_DIR, 'images/val'),
            'test': os.path.join(self.config.OUTPUT_DIR, 'images/test')
        }
        for split_dir in split_dirs.values():
            os.makedirs(split_dir, exist_ok=True)

        splits = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}
        for split_name, df in splits.items():
            print(f"Copying {split_name} images...")
            missing_images = []
            for _, row in df.iterrows():
                img_name = row[self.config.IMAGE_ID_COLUMN]
                src_path = os.path.join(self.config.IMAGE_DIR, img_name)
                dst_path = os.path.join(split_dirs[split_name], img_name)

                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    missing_images.append(img_name)

            print(f"✓ {split_name.capitalize()}: {len(df) - len(missing_images)}/{len(df)} images copied")
            if missing_images:
                print(
                    f"⚠️ Missing: {len(missing_images)} (e.g., {missing_images[:3]}{'...' if len(missing_images) > 3 else ''})")

        print(f"✓ Images split into {split_dirs['train']}, {split_dirs['val']}, {split_dirs['test']}")

    def compute_class_weights(self):
        """Compute class weights"""
        print("\n" + "=" * 50)
        print("STEP 5: CLASS WEIGHTS")
        print("=" * 50)

        class_weights = {}
        for col in self.config.LABEL_COLUMNS:
            pos_count = self.train_df[col].sum()
            neg_count = len(self.train_df) - pos_count
            total = pos_count + neg_count
            weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
            weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0
            class_weights[col] = {0: weight_neg, 1: weight_pos}
            print(f"{col}: pos_weight={weight_pos:.3f}, neg_weight={weight_neg:.3f}")

        self.class_weights = class_weights
        with open(f"{self.config.OUTPUT_DIR}/class_weights.json", 'w') as f:
            json.dump(class_weights, f, indent=2)
        print(f"✓ Class weights saved")

        return class_weights

    def create_augmentation_pipeline(self):
        """Create augmentation pipeline using torchvision"""
        print("\n" + "=" * 50)
        print("STEP 6: DATA AUGMENTATION")
        print("=" * 50)

        class AddGaussianNoise(object):
            def __init__(self, mean=0.0, std=1.0, p=0.2):
                self.mean = mean
                self.std = std
                self.p = p

            def __call__(self, tensor):
                if torch.rand(1) < self.p:
                    noise = torch.randn(tensor.size()) * self.std + self.mean
                    return tensor + noise
                return tensor

        train_transform = transforms.Compose([
            transforms.Resize(self.config.TARGET_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
            AddGaussianNoise(mean=0.0, std=0.05, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(self.config.TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("✓ Augmentation pipelines created:")
        print("   Training: Geometric + Optical")
        print("   Validation/Test: Resize + Normalize")
        print("   Target size:", self.config.TARGET_SIZE)

        return train_transform, val_transform

    def apply_balancing_techniques(self):
        """Suggest balancing techniques"""
        print("\n" + "=" * 50)
        print("STEP 7: CLASS BALANCING")
        print("=" * 50)

        label_counts = {col: self.train_df[col].sum() for col in self.config.LABEL_COLUMNS}
        max_count = max(label_counts.values())
        minority_classes = [col for col, count in label_counts.items() if count < max_count * 0.1]

        print("Techniques:")
        print("1. ✓ Class weights (computed)")
        print("2. Oversampling: Random, SMOTE, or augmentation")
        print("3. Loss: Focal Loss, Weighted BCE")

        if minority_classes:
            print(f"\n⚠️ Minority classes: {minority_classes}")
            print("   Recommendations: Use class weights or oversampling")
        else:
            print("✓ No severe imbalances")

    def generate_preprocessing_summary(self):
        """Generate summary"""
        print("\n" + "=" * 50)
        print("STEP 8: SUMMARY")
        print("=" * 50)

        summary = {
            'dataset_info': {
                'total_samples': len(self.df),
                'label_columns': self.config.LABEL_COLUMNS,
                'target_size': self.config.TARGET_SIZE
            },
            'splits': {
                'train': len(self.train_df),
                'validation': len(self.val_df),
                'test': len(self.test_df)
            }
        }

        print(f"Total samples: {summary['dataset_info']['total_samples']:,}")
        print(
            f"Train: {summary['splits']['train']:,}, Val: {summary['splits']['validation']:,}, Test: {summary['splits']['test']:,}")

        with open(f"{self.config.OUTPUT_DIR}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved")

    def run(self):
        """Execute all steps"""
        print("🚀 Starting Pipeline...")
        try:
            self.load_and_inspect_dataset()
            self.analyze_label_distribution()
            self.inspect_images()
            self.create_data_splits()
            self.compute_class_weights()
            self.create_augmentation_pipeline()
            self.apply_balancing_techniques()
            self.generate_preprocessing_summary()
            print("\n✅ Pipeline completed!")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise


def main():
    config = Config()
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()