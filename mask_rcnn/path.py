import os


ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
DATA_DIR = os.path.join(ROOT_DIR, "data")
BIG_TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
SHIP_CONTAINING_TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train_only_containing_ship")
BIG_TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
SHIP_CONTAINING_TEST_DATA_DIR = os.path.join(DATA_DIR, "test_only_containing_ship")
TEST_TEMP = os.path.join(DATA_DIR, "test_temp")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
DETECTED = os.path.join(ROOT_DIR, "detected")
DETECTED_MASKS_DIR = os.path.join(DETECTED, "mask")
DETECTED_IMAGE_DIR = os.path.join(DETECTED, "image")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2018"

# Path to trained weights file
BALLOON_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_balloon.h5")
COCO_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_coco.h5")
