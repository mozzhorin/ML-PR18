import os
import matplotlib.pyplot as plt
from skimage.io import imread
from utils.data_utils import *


train = os.listdir('./data/train')
print(len(train))

test = os.listdir('./data/test')
print(len(test))

masks = pd.read_csv('./data/train_ship_segmentations.csv')
masks.head()


ImageId = '00113a75c.jpg'

img = imread('./data/train/' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()


# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.savefig("./assets/mask.png")
plt.show()
