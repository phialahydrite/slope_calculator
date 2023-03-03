import pims
from utils import *


## pjt_slope15_prebuilt_062218, high fric baseline parameters
images = pims.ImageSequence('data/*.jpg')
xmin,xmax,ymin,ymax = 95, 5095,300, 1290 # pixel boundaries of image crop
scale=63. #pixels/cm spatial
im_w=images.frame_shape[1]
im_h=images.frame_shape[0]
surfs = 'pjt_slope15_prebuilt_062218_surfnocv_2023.h5'

# calculate deformation front positions 
front = deformation_front_mode(surfs,xmin,ymax,im_h)

# calculate surface slopes
slopes = slope_deffront(surfs,front,plotting=True)
