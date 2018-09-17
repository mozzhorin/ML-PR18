import matplotlib.pyplot as plt
from skimage.draw import polygon
from ship import *

"""
A small test file for length run decoding
"""
np.set_printoptions(threshold=np.inf)
fileDir = os.path.dirname(os.path.realpath('__file__'))
trainDir = os.path.join(fileDir, 'data/train')

train = os.listdir(trainDir)
print(len(train))

testDir = os.path.join(fileDir, 'data/test')
test = os.listdir(testDir)
print(len(test))


extension = "jpg"
train_segments_filename = os.path.join(fileDir, 'data/train_ship_segmentations.csv')
filenames = [f for f in os.listdir(os.path.join(fileDir, 'data/train')) if f.endswith(extension)]
masks = pd.read_csv(train_segments_filename, keep_default_na=False)
masks.head()

ImageId = '00113a75c.jpg'
#ImageId = '0007b8229.jpg'

train_image_filename = os.path.join(fileDir, 'data/train/' + ImageId)
img = imread(train_image_filename)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()


imgp = np.zeros((1536,2048), dtype=np.uint8)
r = np.array([1020,1000,994,1003,1023,1050,1089,1134,1190,1265,1321,1361,1403,1428,1442,1445,1441,1427,1400,1361,1316,1269,1228,1198,1207,1210,1190,1177,1172,1174,1170,1153,1127,1104,1061,1032,1020])
c = np.array([963,899,841,787,738,700,663,638,621,619,643,672,720,765,800,860,896,942,990,1035,1079,1112,1129,1134,1144,1153,1166,1166,1150,1136,1129,1122,1112,1084,1037,989,963])
rr, cc = polygon(r, c)
imgp[rr, cc] = 1

# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768), dtype=np.uint8)
for mask in img_masks:
    all_masks += rle_decode(mask)

#print(all_masks.astype(np.bool))
print(np.ones([all_masks.shape[-1]], dtype=np.int32))

assets_filename = os.path.join(fileDir, 'assets/mask_ship.png')

fig, axarr = plt.subplots(1, 4, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
axarr[3].imshow(imgp)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.savefig(assets_filename)
plt.show()


config = ShipConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH
train(model, config)
