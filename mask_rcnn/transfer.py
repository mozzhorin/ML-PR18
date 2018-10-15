from utils.data_utils import  *
import csv
import pandas as pd
from path import *


## a helper class for data

filter_images_with_masks = False
convert_txt_to_csv = False
copy_files = False
copy_train_data = False
copy_test_data = False
sort_csv = False

train_mask_csv = 'train_ship_segmentations.csv'
test_mask_csv = 'train_ship_segmentations.csv'
txt_file = 'filtered_ships_05_v2.txt'
test_csv_file = 'test_files_with_mask_05.csv'
train_csv_file = 'train_files_with_mask.csv'
submission_csv =  'submission_big_dataset.csv'
sorted_submission_csv = 'sorted_submission_big_dataset.csv'
sorted_csv_file = 'sorted_train_files_with_mask.csv'


## checking the csv file for submission
df = pd.read_csv(os.path.join(DATA_DIR, submission_csv))
df['ImageId'].nunique()
print('number of unique images: ', df['ImageId'].nunique())
pd.set_option("max_r",200)
gr = df.groupby("ImageId")["EncodedPixels"].apply(lambda x: x.isnull().any() and len(x) > 1)
print(gr.value_counts())  # should all be false
print(gr[gr])  # these images have predictions and nan rows


# csv_file_deletion = os.path.join(DATA_DIR, 'submission_deletion.txt')
# with open(csv_file_deletion, 'w') as output_file:
#     print('Filename:', gr[gr], file=output_file)

if sort_csv:
    csv_file = os.path.join(DATA_DIR, submission_csv)
    sorted_csv_file = os.path.join(DATA_DIR, sorted_submission_csv)
    #csv_sorter(csv_file, sorted_csv_file)
    csv_sorter(csv_file, sorted_csv_file, sort_key=0)

if convert_txt_to_csv:
    txt_file = os.path.join(DATA_DIR, txt_file)
    csv_file = os.path.join(DATA_DIR, test_csv_file)
    convertTextToCSV(txt_file, csv_file)


if filter_images_with_masks:
    masks_file_path = os.path.join(DATA_DIR, train_mask_csv)
    masks = pd.read_csv(masks_file_path)
    filenames = load_filenames(BIG_TRAIN_DATA_DIR)
    print(len(filenames))
    print(len(masks))
    images_with_mask = []

    #geting data which contains ship
    for filename in filenames:
        if not is_empty(filename, masks):
            images_with_mask.append(filename)


    print(len(images_with_mask))
    images_with_mask.sort()
    with open(os.path.join(DATA_DIR, train_csv_file), 'w') as file:
        wr = csv.writer(file)
        wr.writerow(images_with_mask)



"""
a helper for copying files from source_folder into the dest_folder 
with references from a CSV file
"""

if copy_files:
    if copy_train_data:
        copy_file_from_source_to_destination_based_on_csv(BIG_TRAIN_DATA_DIR, SHIP_CONTAINING_TRAIN_DATA_DIR, os.path.join(DATA_DIR, train_csv_file))

        
    if copy_test_data:
        copy_file_from_source_to_destination_based_on_csv(BIG_TEST_DATA_DIR, SHIP_CONTAINING_TEST_DATA_DIR, os.path.join(DATA_DIR, test_csv_file))





