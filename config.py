# ODIR/src/config.py
import os

# --- THƯ MỤC GỐC CỦA DỰ ÁN ---
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- ĐƯỜNG DẪN DỮ LIỆU ---
FULL_DF_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "data", "full_df.csv")
IMAGE_BASE_PATH = "D:\\Deep_learning\\proj\\data\\preprocessed_images"

BINARY_DF_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "binary_classification_df.csv")
TRAIN_DF_PATH = os.path.join(BASE_PROJECT_DIR, "train_df.csv")
VAL_DF_PATH = os.path.join(BASE_PROJECT_DIR, "val_df.csv")
TEST_DF_PATH = os.path.join(BASE_PROJECT_DIR, "test_df.csv")

# --- THƯ MỤC LƯU MODEL VÀ KẾT QUẢ ---
MODEL_DIR = os.path.join(BASE_PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(BASE_PROJECT_DIR, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- CẤU HÌNH ẢNH ---
# VGG16 thường được huấn luyện với ảnh kích thước 224x224.
# BinaryDataPreprocessor.py của bạn có tham số target_size (mặc định là 512x512).
# QUAN TRỌNG: Khi bạn khởi tạo BinaryDataPreprocessor trong file train.py,
# bạn cần truyền IMG_SIZE này vào tham số target_size của nó để đảm bảo tính nhất quán.
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
IMG_SHAPE = IMG_SIZE + (3,)

# --- CẤU HÌNH HUẤN LUYỆN ---
BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.001
EPOCHS_TRANSFER_LEARNING = 15 # Có thể bắt đầu với 10-20 epochs
EPOCHS_FINE_TUNING = 10     # Có thể bắt đầu với 10-20 epochs

# --- CẤU HÌNH MODEL ---
BASE_MODEL_NAME = "VGG16" # Chúng ta đã chọn VGG16
# BinaryDataPreprocessor.py của bạn đang dùng preprocess_input của VGG16, điều này là phù hợp.

SAVED_MODEL_NAME_TRANSFER = f"binary_classifier_{BASE_MODEL_NAME}_transfer.keras"
SAVED_MODEL_NAME_FINE_TUNED = f"binary_classifier_{BASE_MODEL_NAME}_fine_tuned.keras"

# Cấu hình cho Fine-tuning VGG16
# VGG16 có các block conv. block5_conv1 (lớp thứ 15 nếu không tính input layer)
# là một điểm tốt để bắt đầu "mở băng" (unfreeze).
# Hãy kiểm tra model.summary() để biết chỉ số lớp chính xác.
FINE_TUNE_AT_LAYER_INDEX = 15 # Chỉ số của lớp block5_conv1 trong base_model.layers
FINE_TUNE_LEARNING_RATE = INITIAL_LEARNING_RATE / 10

# --- CẤU HÌNH CALLBACKS ---
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
MODEL_CHECKPOINT_MONITOR = 'val_auc'
MODEL_CHECKPOINT_MODE = 'max'

# --- TÊN CỘT TRONG FILE CSV ---
FILENAME_COLUMN = 'filename' # Kiểm tra lại tên cột trong file CSV của bạn
LABEL_COLUMN = 'is_abnormal'