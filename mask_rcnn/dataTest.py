from ship import *

config = ShipConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH
train(model, config)
