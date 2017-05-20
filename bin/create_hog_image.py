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

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    '''
        calculates and returns hog features of an image
        img: 1-color-channel image
        orient: amount of orientation bins
        pix_per_cell: width/height of a cell in pixels
        cell_per_block: amount of cells per block
        vis: True if a visualization should also be returned
        feature_vec: True if the hog feature should be returned as a 1D vector
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def writeImage(item, dir, basename, cmap=None):
    '''
        write an image(s) to file
        fig: matplotlib figure or image as an numpy array
        filePath: path to write the image to
    '''
    
    
    # create dir if nonexistent
    if not os.path.isdir(dir):
        log('info', 'creating output directory: ' + dir)
        os.mkdir(dir)
    
    # if numpy array - write it
    #if type == 

    # define filename
    file = dir + '/' + basename + '.png'
    log('info', 'writing image: ' + file)

    # if ndarray
    if isinstance(item, np.ndarray):
        if len(item.shape) == 1:
            fig = plt.figure(1)
            ax = plt.axes()
            plt.plot(item)
            fig.savefig(file)
        else:
            mpimg.imsave(file, item, cmap=cmap)
    else:
        fig = item
        fig.savefig(file)
    
    plt.clf()

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

print('shape image', image.shape)

image_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# # create hog of image
# image_features, hog_image = get_hog_features(image_YCrCb[:,:,0], orient, pix_per_cell, cell_per_block, 
#                         vis=True, feature_vec=False)
# 
# 
# writeImage(hog_image, '.', 'hog_image')

# create histogram of color

channel1_hist = np.histogram(image_YCrCb[:,:,0], bins=32, range=(0, 256))
channel2_hist = np.histogram(image_YCrCb[:,:,1], bins=32, range=(0, 256))
channel3_hist = np.histogram(image_YCrCb[:,:,2], bins=32, range=(0, 256))

print('type of channel1_hist', type(channel1_hist))
print('channel1_hist', channel1_hist)

plt.hist(channel1_hist, bins=32)

plt.show()

writeImage(hog_image, '.', 'histogram_color_image')



