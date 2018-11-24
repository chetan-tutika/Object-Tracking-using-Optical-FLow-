############ Easy ###############################
#NUM_FEATURES = 50
#FEATURES_QUALITY = 0.1
#FEATURES_MIN_DISTANCE = 1
#
#COMBINE = False
#TRANSFORM_COMBINE = 'affine'
#TRANSFORM = 'similarity'
#ERROR_TH = 4
#
#UPDATE_FREQ = 400
#NUM_ITERS_OPTICAL_FLOW = 5
#
#GAUSSIAN_KERNEL_WIDTH = 3
#
#NUM_OBJECTS_TO_TRACK = 2
#################################################



############ MEDIUM ###################

NUM_FEATURES = 50
FEATURES_QUALITY = 0.1
FEATURES_MIN_DISTANCE = 1

COMBINE = False  # to select a combination of affine and similarity
TRANSFORM_COMBINE = 'affine'
TRANSFORM = 'similarity'
ERROR_TH = 4.6

UPDATE_FREQ = 18
NUM_ITERS_OPTICAL_FLOW = 5

GAUSSIAN_KERNEL_WIDTH = 3

NUM_OBJECTS_TO_TRACK = 1

