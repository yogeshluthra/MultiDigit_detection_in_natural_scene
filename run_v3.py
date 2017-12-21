import os
import numpy as np
# import matplotlib.pyplot as plt
# import scipy as sp
# import h5py
# import pandas as pd
import cv2
# import matplotlib.pyplot as plt
# import sklearn
from time import time
import argparse

from PreProcessing import image_preprocessings as PreProcImg
# from PreProcessing import digitStruct
# from utilities import segmentation
from utilities import localization_n_detection as LocDet

# Some pre-reqs and assumptions
input_shape=(32,32,3)
num_classes=11
max_recognizable_digits = 5

# Load Network
from keras.models import load_model
model = load_model('./trained_models/multi_digit_classifier_FullyTrainVGG16_ReducedNetwork_Dataset1_2xCrops_AugmentedWithFalseImages_withRandomRotations.h5')

# Test images
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-r", "--rotation", help = "Rotation to be applied to image")
ap.add_argument("-md", "--minDim_size", help = "max pixel limit on minimum dimension of image (whether width or height)")
ap.add_argument("-n", "--n_expansions", help = "number of box size expansions to be applied")
args = vars(ap.parse_args())

# test_dict = {
#             'house1.jpg': {'rotation': 20, 'minDim_size' : 300, 'n_expansions' : 0},
#             'house3.jpg': {'rotation': -20, 'minDim_size' : 250, 'n_expansions' : 0},
#             'house6.jpg': {'rotation': 0, 'n_expansions' : 0, 'minDim_size' : 400},
#             'house15.jpg': {'rotation': 0, 'n_expansions': 0, 'minDim_size' : 150},
#             'house14.jpg': {'rotation': 0, 'minDim_size' : 150, 'n_expansions' : 0},
#             }

# Digits: Localization and Detection
starttime=time()
test_file = args.get('image')
rotation = args.get('rotation', 0)
minDim_size = args.get('minDim_size', 300)
n_expansions = args.get('n_expansions', 0)
test_image = cv2.imread(test_file, 1)
# - Rotating
if rotation!=0: test_image = PreProcImg.rotate_image(test_image, rotation)
# - Localization
clusters, compressed_image, df_hits = LocDet.localize_digits(model, test_image,
                                                            debug=False,
                                                            minDim_size = minDim_size, # 500 worked ok
                                                            strides = (4, 4), # (2,4) worked ok
                                                            expansion_factor=1.5,
                                                            min_bSize_for_pyramids=32,
                                                            n_expansions=n_expansions, # some issue with expansions
                                                            max_strides_for_colocated_points=2.) # 2 worked ok
# - Detection
test_out_file_path = 'test_out.png'
LocDet.detect_digits(clusters, compressed_image, test_out_file_path, debug=False)
print('\n\n See detection result in test_out.png \n\n')
print('total time taken:',time()-starttime)
