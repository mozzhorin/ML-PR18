import os
import numpy as np

# ref.: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref.: https://www.kaggle.com/inversion/run-length-decoding-quick-star

ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train1")

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




def load_filenames(data_dir):
        if not data_dir:
            data_dir = TRAIN_DATA_DIR
        extension = ".jpg"
        return [f for f in os.listdir(data_dir) if f.endswith(extension)]
