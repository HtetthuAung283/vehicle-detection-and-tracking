#!/usr/bin/python


import os
import sys
import argparse
from time import time
from moviepy.editor import VideoFileClip

# add lib to path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../lib")
from helper_vehicle_detection import *
from detection import Detection


#======================
#
# SETTING PARAMETERS FOR USING THE SAME VALUES FOR LEARNING AND CLASSIFYING
#
#----------------------
# in which color space the feature extraction
color_space = 'LUV'
# spatial size for color histogram
spatial_size = (32, 32)
hist_bins = 32
# amount orientation bins
orient = 9
pix_per_cell = 8
cell_per_block = 2
# which color channels to use for hog features 0|1|2|'ALL'
hog_channel = 'ALL'
# use spatial features
spatial_feat = True
# use histogram color features
hist_feat = True
# use hog features
hog_feat = True

def single_img_features_1(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, feature_vec=True,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    '''
        extract the features from one rgb image (ndarray)
        img: image in rgb color space
        img_hog: the whole image as a 3-channel-hog
    '''
    
#     print('shape img', img.shape)
#     print('min img', np.min(img))
#     print('max img', np.max(img))

    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=feature_vec))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)
        

        #8) Append features to list
        #print('hog_features shape', hog_features.shape)
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def compare_ndarrays(ndarray1, ndarray2):
    ndarray1_flat = ndarray1.flatten()
    ndarray2_flat = ndarray2.flatten()

    if(ndarray1 == ndarray2).all():
        print("ARRAYS ARE EQUAL")
    else:
        print("ARRAYS ARE NOT EQUAL")

    print('the first 5 elements are')
    print('ndarray1: ', ndarray1_flat[0:5])
    print('ndarray2: ', ndarray2_flat[0:5])

# read image
image = mpimg.imread(sys.argv[1])
#image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
#print('image min:', np.min(image))
#print('image max:', np.max(image))
print('shape image', image.shape)

cutout = image[500:564, 800:864] # Crop from y1, y2, x1, x2 -> 200, 400, 100, 300
#print('cutout min:', np.min(cutout))
#print('cutout max:', np.max(cutout))

# create hog of image
image_features = single_img_features_1(image, color_space='LUV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=False, hist_feat=False, hog_feat=True, feature_vec=False)

print('shape image_features', image_features.shape)

# create hog of image
image_features_vec = single_img_features_1(image, color_space='LUV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=False, hist_feat=False, hog_feat=True, feature_vec=True)


print('shape image_features_vec', image_features_vec.shape)

compare_ndarrays(image_features.ravel(), image_features_vec)

# subsample hog
image_features_subsample = image_features[500:564, 800:864]
#print('shape image_features_subsample', image_features_subsample.shape)
#print('shape ravel image_features_subsample', image_features_subsample.ravel().shape)

cutout_features = single_img_features_1(cutout, color_space='LUV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=False, hist_feat=False, hog_feat=True)




#print('shape cutout', cutout.shape)
#print('shape cutout_features', cutout_features.shape)

# plt.imshow(image)
# plt.imshow(cutout)
# plt.show()

# subsample a part of hog image


# create a cutout of image - same as subsampled hog image


# create hog of cutout


# compare subsample with cutout

