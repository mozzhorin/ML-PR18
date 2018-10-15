"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import datetime
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread, imsave
from mrcnn.config import Config
from mrcnn import model as modellib
from utils import utils
from utils.data_utils import *
import path as pt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from timeit import default_timer as timer

##Directories
ROOT_DIR = pt.ROOT_DIR
DATA_DIR = pt.DATA_DIR
TRAIN_DATA_DIR = pt.SHIP_CONTAINING_TRAIN_DATA_DIR
BIG_TRAIN_DATA_DIR = pt.BIG_TRAIN_DATA_DIR
TEST_DATA_DIR = pt.SHIP_CONTAINING_TEST_DATA_DIR
BIG_TEST_DATA_DIR = pt.BIG_TEST_DATA_DIR
ASSETS_DIR = pt.ASSETS_DIR
# Directory to save logs and trained model
MODEL_DIR = pt.MODEL_DIR
DETECTED = pt.DETECTED
DETECTED_MASKS_DIR = pt.DETECTED_MASKS_DIR
DETECTED_IMAGE_DIR = pt.DETECTED_IMAGE_DIR

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = pt.DEFAULT_LOGS_DIR
DEFAULT_DATASET_YEAR = pt.DEFAULT_DATASET_YEAR

# Path to trained weights file
BALLOON_WEIGHTS_PATH = pt.BALLOON_WEIGHTS_PATH
COCO_WEIGHTS_PATH = pt.COCO_WEIGHTS_PATH

############################################################
#  Configurations
############################################################

class ShipConfig(Config):
    """Configuration for training on the ship dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ship"

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + ship

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 90000

    VALIDATION_STEPS = 10000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.90

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (256, 256)  # (height, width) of the mini-mask

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

############################################################
#  Dataset
############################################################

class ShipDataset(utils.Dataset):

    def __init__(self, aug=False, mode="train"):
        utils.Dataset.__init__(self)
        self.filenames = []
        self.aug = aug
        self.mode = ''
        if mode == 'train':
            self.mode = 'train'
            masks_file_path = os.path.join(DATA_DIR, 'train_ship_segmentations.csv')
            self.masks = pd.read_csv(masks_file_path, keep_default_na=False)

        elif mode == "test":
            self.mode = 'test'
            #masks_file_path = os.path.join(DATA_DIR, 'test_ship_segmentations.csv')
            #self.masks = pd.read_csv(masks_file_path, keep_default_na=False)

        self.len = len(self.filenames)

    def load_ship(self, dataset_dir):
        """Load a subset of the ship dataset.
        dataset_dir: Root directory of the dataset.
        """

        # Add classes. We have only one class to add.
        self.add_class("ship", 1, "ship")

        for filename in self.filenames:
            masks = self.masks.loc[self.masks['ImageId'] == filename, 'EncodedPixels'].tolist()
            image_path = os.path.join(dataset_dir, filename)
            height, width = 768, 768
            masks = list(filter(None, masks))
            self.add_image(
                "ship",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask=masks)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a ship dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        img_masks = self.masks.loc[self.masks['ImageId'] == info["id"], 'EncodedPixels'].tolist()

        # Take the individual ship masks and create a single mask array for all ships
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        all_masks = np.zeros([info["height"], info["width"], len(img_masks)], dtype=np.uint8)

        if img_masks == [-1]:
            return all_masks

        for idx, value in enumerate(img_masks):
            mask += rle_decode(value)
            all_masks[..., idx] = mask

        return all_masks.astype(np.bool), np.ones([all_masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ship":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, config, dataset=TRAIN_DATA_DIR):
    """Train the model."""

    dataset_train, dataset_val = load_dataset(mode="train", dataset_path=dataset)

    ## Based on the context, we need the full layer training.
    # starting from COCO trained weights,
    print("Training the whole network and its layers")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='all')

## load data from provided path
def load_dataset(mode="train", dataset_path=TRAIN_DATA_DIR):
    train_dataset_filenames = load_filenames(dataset_path)
    train_dataset_filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    np.random.seed(230)
    np.random.shuffle(train_dataset_filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
    if mode == "train":
        train_len = int(0.9*len(train_dataset_filenames))
        valid_len = len(train_dataset_filenames) - train_len
    else:
        train_len = len(train_dataset_filenames)

    dataset_train_filenames = train_dataset_filenames[:train_len]
    dataset_val_filenames = train_dataset_filenames[train_len:]
    print(len(dataset_train_filenames))
    print(len(dataset_val_filenames))

    # Training dataset.
    dataset_train = ShipDataset()
    dataset_train.filenames.extend(dataset_train_filenames)
    dataset_train.load_ship(dataset_path)
    dataset_train.prepare()

    if mode == "train":
        # Validation dataset
        dataset_val = ShipDataset()
        dataset_val.filenames.extend(dataset_val_filenames)
        dataset_val.load_ship(dataset_path)
        dataset_val.prepare()

    else:
        dataset_val = []


    return  dataset_train, dataset_val

##save detected objects together with their masks as an image
def save_detected_image(image, masks, test_image_filename):
    # Save output
    file_name = test_image_filename[:-4] + "_detected_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    dir_to_save_image = os.path.join(DETECTED_IMAGE_DIR, file_name)
    imsave(dir_to_save_image, image)
    for idx in range(masks.shape[-1]):
        #Save mask output
        mask_file_name =  test_image_filename[:-4] + "_mask_" + str(idx) + "_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        dir_to_save_mask = os.path.join(DETECTED_MASKS_DIR, mask_file_name)
        plt.imsave(dir_to_save_mask, masks[...,idx], cmap=cm.gray)
        #imsave(dir_to_save_mask, masks[...,idx].astype(np.uint8))


##apply color onto an image based on its mask
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = gray2rgb(rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

##detection
def detect_and_color_splash(model, image_path=None, video_path=None, submission_file_name_extender=None, save_to_file=False, multi_entry_submission=False):
    assert image_path or video_path
    # Image or video?
    masks_dict = {}
    masks_rle = []
    start = timer()
    seconds_elapsed = 0
    i = 0
    k = 0
    if image_path:
        test_image_filenames = load_filenames(image_path)
        # Run model detection and generate the color splash effect
        #print("Running on {}".format(args.image))
        print("Running on {}".format(image_path))
        print("Image Count for detection", len(test_image_filenames))
        for test_image_filename in test_image_filenames:
            # Read image
            #image = imread(args.image)
            image = imread(os.path.join(image_path, test_image_filename))
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            print("image file name", test_image_filename)
            ## very memory hungry
            if submission_file_name_extender is not None:
                masks_dict[test_image_filename] = r['masks']
                i += 1
                k += 1
                print("counter: ", k)
                if i == 43000:
                    end = timer()
                    seconds_elapsed += end - start
                    print("detection duration(in Sec) after " + str(i) + " images : ", end - start)
                    if multi_entry_submission:
                        masks_rle += gather_mask_rle(masks_dict)
                    else:
                        masks_rle += gather_mask_rle_alt(masks_dict)

                    masks_dict = {}
                    i = 0
                    gc.collect()
                    start = timer()
            if save_to_file:
                #Color splash
                splash = color_splash(image, r['masks'])
                save_detected_image(splash, r['masks'], test_image_filename)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    #print("Saved to ", file_name)
    end = timer()
    seconds_elapsed += end - start
    print("detection duration(in Sec) of " + str(k) + " images: ", seconds_elapsed)
    if submission_file_name_extender is not None:
        if masks_dict:
            if multi_entry_submission:
                masks_rle += gather_mask_rle(masks_dict)
            else:
                masks_rle += gather_mask_rle_alt(masks_dict)

        create_submission_file(masks_rle, submission_file_name_extender=submission_file_name_extender)



############################################################
#  Training
############################################################

def init_training(mode, dataset, logs=DEFAULT_LOGS_DIR, weights="last", image=TEST_DATA_DIR, video=None, submission_file_name_extender=None, save_to_file=False):
    # Validate arguments
    if mode == "train":
        assert dataset, "Argument --dataset is required for training"
    elif mode == "splash":
        assert image or video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", weights)
    if mode == "train":
        print("Dataset: ", dataset)
    else:
        print("Dataset: ", image)
    print("Logs: ", logs)
    print("Mode: ", mode)

    # Configurations
    if mode == "train":
        config = ShipConfig()
    else:
        class InferenceConfig(ShipConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if mode == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs)
    elif mode == "splash":
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if mode == "train":
        train(model, config, dataset)
    elif mode == "splash":
        detect_and_color_splash(model, image_path=image,
                                video_path=video, submission_file_name_extender=submission_file_name_extender, save_to_file=save_to_file)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(mode))





## init train or detection via CLI
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ships on satellite images .')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/ship/dataset/",
                        help='Directory of the ship dataset for training the model')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ShipConfig()
    else:
        class InferenceConfig(ShipConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

