from config import Config
import numpy as np


class BowlConfig(Config):
    NAME = "Bowl"


    # todo use resnetxt
    BACKBONE = "resnet101"

    # Train on 1 GPU and 16 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # change it when use different model
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    NUM_CLASSES = 1 + 1

    # change it for bowl data
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # change when training and testing
    RPN_NMS_THRESHOLD = 0.3

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 84

    MEAN_PIXEL = np.array([43.5, 39.6, 48.2])

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 256
    # IMAGE_MAX_DIM = 256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 500

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
