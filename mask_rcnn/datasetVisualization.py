import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


from utils import utils, data_utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import ship
from ship import ASSETS_DIR
import balloon
#from balloon import ROOT_DIR, DATA_DIR


# Root directory of the project

ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train1")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
#Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask R-CNN
#sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2018"

# Path to trained weights file
BALLOON_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_balloon.h5")
COCO_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_coco.h5")

##ship data
config = ship.ShipConfig()
SHIP_DIR = TRAIN_DATA_DIR

train_dataset_filenames = data_utils.load_filenames(SHIP_DIR)
train_dataset_filenames.sort()  # make sure that the filenames have a fixed order before shuffling
np.random.seed(230)
np.random.shuffle(train_dataset_filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
train_len = int(0.3*len(train_dataset_filenames))
valid_len = len(train_dataset_filenames) - train_len

dataset_train_filenames = train_dataset_filenames[:train_len]
dataset_val_filenames = train_dataset_filenames[train_len:]

print(train_len)
print(valid_len)

# Load dataset
dataset = ship.ShipDataset()
dataset.filenames.extend(dataset_train_filenames)
dataset.load_ship(SHIP_DIR)
dataset.prepare()


print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# config = balloon.BalloonConfig()
# BALLOON_DIR = DATA_DIR
#
# dataset = balloon.BalloonDataset()
# dataset.load_balloon(BALLOON_DIR, "train")
#
# # Must call before using the dataset
# dataset.prepare()
#
# print("Image Count: {}".format(len(dataset.image_ids)))
# print("Class Count: {}".format(dataset.num_classes))
# for i, info in enumerate(dataset.class_info):
#     print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for idx, image_id in enumerate(image_ids):
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, img_idx=idx)


# Load random image and mask.
for idx, image_id in enumerate(image_ids):
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)
    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, img_idx=idx)


# # Load random image and mask.
# for idx, image_id in enumerate(image_ids):
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     original_shape = image.shape
#     # Resize
#     image, window, scale, padding, _ = utils.resize_image(
#         image,
#         min_dim=config.IMAGE_MIN_DIM,
#         max_dim=config.IMAGE_MAX_DIM,
#         mode=config.IMAGE_RESIZE_MODE)
#     mask = utils.resize_mask(mask, scale, padding)
#     # Compute Bounding box
#     bbox = utils.extract_bboxes(mask)
#     # Display image and additional stats
#     print("image_id: ", image_id, dataset.image_reference(image_id))
#     print("Original shape: ", original_shape)
#     log("image", image)
#     log("mask", mask)
#     log("class_ids", class_ids)
#     log("bbox", bbox)
#     # Display image and instances
#     visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, img_idx=idx)

# Generate Anchors
backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                         config.RPN_ANCHOR_RATIOS,
                                         backbone_shapes,
                                         config.BACKBONE_STRIDES,
                                         config.RPN_ANCHOR_STRIDE)

# Print summary of anchors
num_levels = len(backbone_shapes)
anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
print("Count: ", anchors.shape[0])
print("Scales: ", config.RPN_ANCHOR_SCALES)
print("ratios: ", config.RPN_ANCHOR_RATIOS)
print("Anchors per Cell: ", anchors_per_cell)
print("Levels: ", num_levels)
anchors_per_level = []
for l in range(num_levels):
    num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
    anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
    print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))

## Visualize anchors of one cell at the center of the feature map of a specific level

# Load and draw random image
for idx, image_id in enumerate(image_ids):
    image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    levels = len(backbone_shapes)

    for level in range(levels):
        colors = visualize.random_colors(levels)
        # Compute the index of the anchors at the center of the image
        level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
        level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
        print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
                                                                      backbone_shapes[level]))
        center_cell = backbone_shapes[level] // 2
        center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
        level_center = center_cell_index * anchors_per_cell
        center_anchor = anchors_per_cell * (
                (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) \
                + center_cell[1] / config.RPN_ANCHOR_STRIDE)
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                                  edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
            ax.add_patch(p)

        anchor_img_name = "anchor_img" + str(idx) + ".png"
        plt.savefig(os.path.join(ASSETS_DIR, anchor_img_name))


# Create data generator
random_rois = 2000
g = modellib.data_generator(
    dataset, config, shuffle=True, random_rois=random_rois,
    batch_size=4,
    detection_targets=True)

# Get Next Image
if random_rois:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)

    log("rois", rois)
    log("mrcnn_class_ids", mrcnn_class_ids)
    log("mrcnn_bbox", mrcnn_bbox)
    log("mrcnn_mask", mrcnn_mask)
else:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)

log("gt_class_ids", gt_class_ids)
log("gt_boxes", gt_boxes)
log("gt_masks", gt_masks)
log("rpn_match", rpn_match, )
log("rpn_bbox", rpn_bbox)
image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
print("image_id: ", image_id, dataset.image_reference(image_id))

# Remove the last dim in mrcnn_class_ids. It's only added
# to satisfy Keras restriction on target shape.
mrcnn_class_ids = mrcnn_class_ids[:,:,0]


b = 0

# Restore original image (reverse normalization)
sample_image = modellib.unmold_image(normalized_images[b], config)

# Compute anchor shifts.
indices = np.where(rpn_match[b] == 1)[0]
refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
log("anchors", anchors)
log("refined_anchors", refined_anchors)

# Get list of positive anchors
positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
print("Positive anchors: {}".format(len(positive_anchor_ids)))
negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
print("Negative anchors: {}".format(len(negative_anchor_ids)))
neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

# ROI breakdown by class
for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
    if n:
        print("{:23}: {}".format(c[:20], n))

# Show positive anchors
fig, ax = plt.subplots(1, figsize=(16, 16))
visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
                     refined_boxes=refined_anchors, ax=ax)

positive_anchor_image_name = "pos_ancher_img.png"
plt.savefig(os.path.join(ASSETS_DIR, positive_anchor_image_name))

visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids], img_name="negative_anchors")

# Show neutral anchors. They don't contribute to training.
visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)], img_name="neutral_anchors")

if random_rois:
    # Class aware bboxes
    bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]

    # Refined ROIs
    refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:,:4] * config.BBOX_STD_DEV)

    # Class aware masks
    mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

    visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)

    # Any repeated ROIs?
    rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
    _, idx = np.unique(rows, return_index=True)
    print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))



if random_rois:
    # Dispalay ROIs and corresponding masks and bounding boxes
    ids = random.sample(range(rois.shape[1]), 8)

    images = []
    titles = []
    for i in ids:
        image = visualize.draw_box(sample_image.copy(), rois[b,i,:4].astype(np.int32), [255, 0, 0])
        image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
        images.append(image)
        titles.append("ROI {}".format(i))
        images.append(mask_specific[i] * 255)
        titles.append(dataset.class_names[mrcnn_class_ids[b,i]][:20])

    display_images(images, titles, cols=4, cmap="Blues", interpolation="none", img_name="random_rois")

# Check ratio of positive ROIs in a set of images.
if random_rois:
    limit = 10
    temp_g = modellib.data_generator(
        dataset, config, shuffle=True, random_rois=10000,
        batch_size=1, detection_targets=True)
    total = 0
    for i in range(limit):
        _, [ids, _, _] = next(temp_g)
        positive_rois = np.sum(ids[0] > 0)
        total += positive_rois
        print("{:5} {:5.2f}".format(positive_rois, positive_rois/ids.shape[1]))
    print("Average percent: {:.2f}".format(total/(limit*ids.shape[1])))