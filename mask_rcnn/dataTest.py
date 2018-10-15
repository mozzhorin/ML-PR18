from ship import *

##learning
#init_training("train", BIG_TRAIN_DATA_DIR, DEFAULT_LOGS_DIR, "coco")

## continue learning where it was left off
#init_training("train", BIG_TRAIN_DATA_DIR, DEFAULT_LOGS_DIR, "last")

##detection
init_training("splash", BIG_TRAIN_DATA_DIR, DEFAULT_LOGS_DIR, "coco", image=BIG_TEST_DATA_DIR, submission_file_name_extender="big_dataset")

