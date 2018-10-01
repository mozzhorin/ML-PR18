from utils.data_utils import  *
import csv
import pandas as pd
from path import *

"""
a helper for copying files from source_folder into the dest_folder 
with references from a CSV file
"""

copy_file_from_source_to_destination_based_on_csv(BIG_TEST_DATA_DIR, SHIP_CONTAINING_TEST_DATA_DIR, os.path.join(DATA_DIR, "test_files_with_mask.csv"))

copy_file_from_source_to_destination_based_on_csv(BIG_TEST_DATA_DIR, SHIP_CONTAINING_TRAIN_DATA_DIR, os.path.join(DATA_DIR, "train_files_with_mask.csv"))



train_mask_csv = 'train_ship_segmentations.csv'
test_mask_csv = 'test_ship_segmentations.csv'
masks_file_path = os.path.join(DATA_DIR, train_mask_csv)
masks = pd.read_csv(masks_file_path)
filenames = load_filenames(BIG_TEST_DATA_DIR)
print(len(filenames))
print(len(masks))
images_with_mask = []

#geting data which contains ship
for filename in filenames:
    if not is_empty(filename, masks):
        images_with_mask.append(filename)


print(len(images_with_mask))
with open(os.path.join(DATA_DIR, 'test_files_with_mask_new.csv'), 'w') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(images_with_mask)




