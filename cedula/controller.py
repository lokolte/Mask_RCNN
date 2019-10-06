import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from custom import ObjectInference

if __name__ == '__main__':

    FLAG_TRAINING=True # Change to false to just train

    modelToTrain = ObjectInference()

    # Train a new model starting from pre-trained COCO weights
    # dataset=/path/to/balloon/dataset weights=coco

    # Resume training a model that you had trained earlier
    # dataset=/path/to/balloon/dataset weights=last

    # Train a new model starting from ImageNet weights
    # dataset=/path/to/balloon/dataset weights=imagenet

    if FLAG_TRAINING:
        modelToTrain.trainModel(dataset='/home/alforro/TigoCampusParty/Mask_RCNN/cedula/images', weights='coco')

    # Apply color splash to an image
    # weights=/path/to/weights/file.h5 image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    # weights=last video=<URL or path to file>

    modelToTrain.splashModel(weights='last', image='/Users/jesusaguilar/projects/git/Mask_RCNN/cedula/images/splash/cedula_1.jpeg')