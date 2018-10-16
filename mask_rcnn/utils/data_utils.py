#import os
from path import *
import numpy as np
import pandas as pd
import csv, operator, random
from shutil import copy
from skimage.morphology import label, binary_opening, disk
import gc; gc.enable() # memory is tight

def multi_rle_encode(img):
    labels = label(img[:, :])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ref.: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref.: https://www.kaggle.com/inversion/run-length-decoding-quick-star


def rle_decode(mask_rle, shape=(768, 768)):
    """"
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    if mask_rle:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction

## gathers jpg image files from the provided folder
def load_filenames(data_dir):
        if not data_dir:
            data_dir = SHIP_CONTAINING_TRAIN_DATA_DIR
        extension = ".jpg"
        return [f for f in os.listdir(data_dir) if f.endswith(extension)]

##gather run-length masks for submission, each image has multiple entry in submission data
def gather_mask_rle(masks_dict):
    out_pred_rows = []
    for file_name, masks in masks_dict.items():
        if masks.size > 0:
            for idx in range(masks.shape[-1]):
                binary_mask = binary_opening(masks[...,idx])
                rle_masks = multi_rle_encode(binary_mask)
                if len(rle_masks)>0:
                    for rle_mask in rle_masks:
                        out_pred_rows += [{'ImageId': file_name, 'EncodedPixels': rle_mask}]
                else:
                    out_pred_rows += [{'ImageId': file_name, 'EncodedPixels': None}]
                gc.collect()
        else:
            out_pred_rows += [{'ImageId': file_name, 'EncodedPixels': None}]

    return out_pred_rows

##gather run-length masks for submission, each image has a single entry in submission data
def gather_mask_rle_alt(masks_dict):
    out_pred_rows = []
    for file_name, masks in masks_dict.items():
        if masks.size > 0:
            mask_complete = np.zeros((768, 768), dtype=np.bool)
            for idx in range(masks.shape[-1]):
                mask_complete = np.logical_or(masks[...,idx], mask_complete)

            binary_mask = binary_opening(mask_complete)
            rle_masks = rle_encode(binary_mask)
            out_pred_rows += [{'ImageId': file_name, 'EncodedPixels': rle_masks}]
        else:
            out_pred_rows += [{'ImageId': file_name, 'EncodedPixels': None}]

    return out_pred_rows

def create_submission_file(rle_data, submission_file_name_extender="ready_for_submission"):
    if submission_file_name_extender is None:
        submission_file_name_extender="ready_for_submission"
    submission_df = pd.DataFrame(rle_data)[['ImageId', 'EncodedPixels']]
    submission_file_name = os.path.join(DATA_DIR, "submission_" + submission_file_name_extender +".csv")
    submission_df.to_csv(submission_file_name, index=False)


##copy files from source to destination based on csv data
def copy_file_from_source_to_destination_based_on_csv(source_folder, destination_folder, ref_csv_filename):
    ##csv has only one line
    pathdict = {}
    with open(ref_csv_filename) as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        a = 0
        for row in filereader:
            pathdict[a] = ','.join(row)
        files = pathdict[a].split(",")
        csvfile.close()

    print(len(files))
    ## use for limiting the data count
    files_temp = random.sample(files, 10000)
    print(len(files_temp))
    for file in files:
        copy(os.path.join(source_folder, file), destination_folder)

def mask_part(pic):
    '''
    Function that encodes mask for single ship from .csv entry into numpy matrix
    '''
    back = np.zeros(768**2)
    starts = pic.split()[0::2]
    lens = pic.split()[1::2]
    for i in range(len(lens)):
        back[(int(starts[i])-1):(int(starts[i])-1+int(lens[i]))] = 1
    return np.reshape(back, (768, 768, 1))

def is_empty(filename, masks):
    '''
    Function that checks if there is a ship in image
    '''
    df = masks[masks['ImageId'] == filename].iloc[:,1]
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]):
        return True
    else:
        return False

def masks_all(filename, marks):
    '''
    Merges together all the ship markers corresponding to a single image
    '''
    df = marks[marks['ImageId'] == filename].iloc[:,1]
    masks= np.zeros((768,768,1))
    if is_empty(filename, marks):
        return masks
    else:
        for i in range(len(df)):
            masks += mask_part(df.iloc[i])
        return np.transpose(masks, (1,0,2))

def convertTextToCSV(txt_file, csv_file):
    with open(txt_file, 'r') as input_file, open(csv_file, 'w') as output_file:
        stripped = sorted((line.strip() for line in input_file))
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(stripped)

def csv_sorter(csv_file, csv_file_sorted, sort_key=None):
    with open(csv_file, 'r') as input_file, open(csv_file_sorted, 'w') as output_file:
        csv_data = csv.reader(input_file, delimiter=',')
        ## sort_key is the index of column to be sorted
        if sort_key is not None:
            csv_data_sorted = sorted(csv_data, key=operator.itemgetter(sort_key))
        else:
            csv_data_sorted = sorted(csv_data)
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(['ImageId', 'EncodedPixels'])
        for row in csv_data_sorted:
            writer.writerow(row)

