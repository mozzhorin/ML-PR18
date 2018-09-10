import numpy as np
import pandas as pd

# ref.: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref.: https://www.kaggle.com/inversion/run-length-decoding-quick-star


def rle_decode(mask_rle, shape=(768, 768)):
    """"
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
