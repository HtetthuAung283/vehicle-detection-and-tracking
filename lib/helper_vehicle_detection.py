import os
from time import time
import datetime
# import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from vehicle import Vehicle
from position import Position


# helper functions
def log(stage, msg):
    '''
        function to print out logging statement in this format:
        
        format
        <time> : <stage> : <msg>
        
        example:
        2017-04-28 12:48:45 : info : chess board corners found in image calibration20.jpg
    '''
    
    
    print(str(datetime.datetime.now()).split('.')[0] + " : " + stage + " : " + msg)

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

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def detect_vehicles(img, detection, clf, X_scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, x_start_stop=[None, None], y_start_stop=[None, None], subsampling=False, retNr=False, format='normal'):
    
        # store for the intermediate steps of the processing pipeline
    imageBank = {}
    imageBank[0] = img

    if retNr is 0:
        return img, detection
    
    # overlap_cells_per_step
    # the cells are made from 8 pixels,
    # the step from one window to another is measured in cells (or in overlay)
    overlap_to_cells_per_step = {0.825: 1,
                                 0.75: 2,
                                 0.625: 3,
                                 0.5: 4,
                                 0.375: 5,
                                 0.25: 6,
                                 0.125: 7,
                                 0: 9
                                 }
    
    overlap = 0.75
    
    # calculate every window anew
    if subsampling == False:
        windows_big = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=(110, 110), xy_overlap=(0.75, 0.75))
        windows_big2 = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=(90, 90), xy_overlap=(0.75, 0.75))
        windows_med = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=(64, 64), xy_overlap=(0.75, 0.75))
        windows_med2 = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=(50, 50), xy_overlap=(0.75, 0.75))
#        windows_sma = slide_window(img, x_start_stop=x_start_stop, y_start_stop=(y_start_stop[0], y_start_stop[0] + (y_start_stop[1] - y_start_stop[0]) / 2), xy_window=(32, 32), xy_overlap=(0.5, 0.5))
        
        hot_windows = search_windows(img, windows_med+windows_big2+windows_big, clf, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    # use subsampling to speed up the calculation
    # subsampling does not work
    else:
        scale =1
        windowsize = [64]
        hot_windows = find_cars(img, windowsize[0], color_space, y_start_stop[0], y_start_stop[1], overlap_to_cells_per_step[0.75], scale, clf, X_scaler, hog_channel, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    tmp_img = np.copy(img)
#    tmp_img = draw_boxes(tmp_img, windows_med2, color=(255, 0, 0), thick=2)
    tmp_img = draw_boxes(tmp_img, hot_windows, color=(0, 255, 0), thick=2)
#    tmp_img = draw_boxes(tmp_img, windows_big, color=(255, 0, 255), thick=2)
#    tmp_img = draw_boxes(tmp_img, hot_windows, color=(255, 255, 255), thick=2)

    imageBank[1] = tmp_img
    if retNr is 1:
        return tmp_img, detection

    # create an empty heatmap
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    heatmap = np.clip(heat, 0, 255)


    # Visualize the heatmap when displaying    
    heatmap = np.clip(heatmap, 0, 255)

    heatmap = heatmap * (255 / np.max(heatmap))
    heatmap = heatmap.astype(int)
    heatmap_as_rgb = np.zeros_like(img)
    heatmap_as_rgb[:,:,0] = heatmap
    heatmap_as_rgb[:,:,1] = heatmap
    heatmap_as_rgb[:,:,2] = heatmap
    
    imageBank[2] = heatmap_as_rgb
    if retNr is 2:
        return heatmap_as_rgb, detection

    # Apply threshold to help remove false positives
    t_heat = apply_threshold(heat, 2)

    # Find final boxes from heatmap using label function
    labels = label(t_heat)
    labels_img = np.copy(labels[0])
    labels_img = labels_img * (255 / np.max(labels_img))
    labels_img = labels_img.astype(int)
    labels_img_as_rgb = np.zeros_like(img)
    labels_img_as_rgb[:,:,0] = labels_img
    labels_img_as_rgb[:,:,1] = labels_img
    labels_img_as_rgb[:,:,2] = labels_img
    
    imageBank[3] = labels_img_as_rgb
    if retNr is 3:
        return labels_img_as_rgb, detection
    

    
    
    draw_img = np.copy(img)

    positions = labels2positions(labels)
    
    detection.addPositions(positions)
    detection.detect()

#    result = draw_labeled_bboxes(draw_img, labels)

    result = draw_boxes(draw_img, detection.getVehicleBoundingBoxes(), color=(0, 255, 0), thick=2)

    imageBank[4] = result

    if format == 'collage4':
        return genCollage(4, imageBank), detection
    
    return result, detection
    
#     draw_img = draw_boxes(draw_img, windows_med2, color=(255, 0, 0), thick=2)
#     draw_img = draw_boxes(draw_img, windows_med, color=(0, 255, 0), thick=2)
#     draw_img = draw_boxes(draw_img, windows_big, color=(255, 0, 255), thick=2)
#     draw_img = draw_boxes(draw_img, hot_windows, color=(255, 255, 255), thick=2)
#     return draw_img

def genCollage(amount, imageBank):
    '''
        generating a 2x2 or 3x3 collage
        amount: 4 -> 2x2 collage; 9 _> 3x3 collage
        return: imageCollage
    '''
    resultImage = None
    
    if amount == 4:
        row1 = cv2.hconcat((imageBank[1], imageBank[2]))
        row2 = cv2.hconcat((imageBank[3], imageBank[4]))
        resultImage = cv2.vconcat((row1, row2))
        resultImage = cv2.resize(resultImage, (1920, int((1920/resultImage.shape[1]) * resultImage.shape[0])))
    
    return resultImage

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def labels2positions(labels):
    '''
        converts the labels into positions
    '''
    
    positions = []
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        myposition = Position( (np.min(nonzerox) + np.max(nonzerox)) / 2, (np.min(nonzeroy) + np.max(nonzeroy)) / 2, np.max(nonzeroy) - np.min(nonzeroy), np.max(nonzerox) - np.min(nonzerox) )
        positions.append(myposition)
        
    return positions

def draw_labeled_bboxes(img, labels):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 1)
        # write text over box
        cv2.putText(img, 'Car', bbox[0], font, 1, (0, 255, 0), 1)
    # Return the image
    return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, window, color_space, ystart, ystop, cells_per_step, scale, svc, X_scaler, hog_channel, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
#    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    
    # convert to desired color space
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img)      

    # scale?
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # deparate channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    #window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    
    hot_windows = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            if hog_channel == 0:
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = hog_feat1
            elif hog_channel == 'ALL':
            # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))


            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                hot_windows.append( ( ( xpos, ypos ), ( xpos + nblocks_per_window, ypos + nblocks_per_window) ) )
                
#             if test_prediction == 1:
#                 xbox_left = np.int(xleft*scale)
#                 ytop_draw = np.int(ytop*scale)
#                 win_draw = np.int(window*scale)
#                 cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
#     return draw_img
    return hot_windows

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, X_scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

   #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        
        
        #5) Scale extracted features to be fed to classifier
        #test_features = scaler.transform(np.array(features).reshape(1, -1))
        test_features = X_scaler.transform(features)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imageFiles, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
        extract the features from a list of images (full path-file-names)
        and return them as a list of feature vectors
    '''
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file in imageFiles:
        file_features = []
        # Read in each one by one directly in RGB space and 0-1 scale
#        image = mpimg.imread(file)
        
        # read image in BGR colorspace and 0-255 scale
        bgr = cv2.imread(file)

        # convert to RGB color space
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        file_features = single_img_features(rgb, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        
        features.append(file_features)
    
    return features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
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
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        

        #8) Append features to list
        #print('hog_features shape', hog_features.shape)
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


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



def createClassifier(mlDir, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    '''
        function to create a trained classifier of vehicles in images
        
        1) if a trained classifier already exists in mlDir, then this classifier should be loaded and returned
        1) determining all filePaths of vehicle- and non-verhicle-images
        2) calculating HOG (Histogram of Oriented Gradients)
        3) calculating color histogram (histogram of colors)
        4) create a single feature-vectors from 2) and 3)
        5) train a linear classifier on 4)
        6) save classifier for later use
        7) return classifier
        
        mlDir: directory with vehicle and non-vehicle images
        return: classifier: trained classifier
    '''

    ###############################
    #
    # STEP 1: READ PRECALCULATED CLASSIFIER IF EXISTS
    #
    ###############################
    # classifier save file
    classifierPkl = mlDir + '/.classifier.pkl'
    allPreTrainDataSets = []
    
    # if one exists, then the classifier can be loaded from there instead of running the whole new training process
    if os.path.isfile(classifierPkl):
        log('info', 'some precalculated classifiers found - loading that: ' + classifierPkl)
        allPreTrainDataSets = pickle.load(open(classifierPkl, "rb"))
        
        for preDataSet in allPreTrainDataSets:
            if preDataSet['color_space'] == color_space:
                if preDataSet['spatial_size'] == spatial_size:
                    if preDataSet['hist_bins'] == hist_bins:
                        if preDataSet['orient'] == orient:
                            if preDataSet['pix_per_cell'] == pix_per_cell:
                                if preDataSet['cell_per_block'] == cell_per_block:
                                    if preDataSet['hog_channel'] == hog_channel:
                                        if preDataSet['spatial_feat'] == spatial_feat:
                                            if preDataSet['hist_feat'] == hist_feat:
                                                if preDataSet['hog_feat'] == hog_feat:
                                                    log('info', 'classifier with exact the same parameters found')
                                                    log('info', str(preDataSet))

                                                    return preDataSet
                                                    #return preDataSet['clf'], preDataSet['X_scaler']
        
        log('info', 'no precalculated classifier - loading that: ' + classifierPkl)

    ###############################
    #
    # STEP 2: DETERMINING ALL FILEPATHS OF VEHICLE- AND NON-VEHICLE IMAGES
    #
    ###############################
    imagesVehicles = getAllFilesFromDirTree(mlDir + '/vehicles')
    imagesNonVehicles = getAllFilesFromDirTree(mlDir + '/non-vehicles')
    log('info', 'determining paths for training/testing images. ' + str(len(imagesVehicles) + len(imagesNonVehicles)) + ' files found.')
    
    ###############################
    #
    # STEP 3: READ IMAGE, GENERATE FEATURES AND LABELS
    #
    ###############################
    log('info', 'reading ' + str(len(imagesVehicles)) + ' images of vehicles and converting them into feature vectors.')

    featureVectorsVehicles = extract_features(imagesVehicles, color_space=color_space, spatial_size=spatial_size,
                                            hist_bins=hist_bins, orient=orient,
                                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    log('info', 'reading ' + str(len(imagesNonVehicles)) + ' images of non-vehicles and converting them into feature vectors.')

    featureVectorsNonVehicles = extract_features(imagesNonVehicles, color_space=color_space, spatial_size=spatial_size,
                                            hist_bins=hist_bins, orient=orient,
                                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((featureVectorsVehicles, featureVectorsNonVehicles)).astype(np.float64)                        
    # Fit a per-column scaler
    log('info', 'scaling feature vectors with a standard scaler.')
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(featureVectorsVehicles)), np.zeros(len(featureVectorsNonVehicles))))
    
    
    ###############################
    #
    # STEP 4: SPLIT TRAINING AND TEST SETS
    #
    ###############################
    # Split up data into randomized training and test sets
    #rand_state = np.random.randint(0, 100)
    log('info', 'randomly split the data in training and test sets.')
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=12)

    log('info', 'Feature vector length:' + str(len(X_train[0])))

    ###############################
    #
    # STEP 5: CREATE AND TRAIN A SVC (Support Vector Machine Classifier)
    #
    ###############################

    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time()
    log('info', 'training SVC...')
    clf = svc.fit(X_train, y_train)
    t2 = time()
    log('info', str(round(t2-t, 2)) + ' seconds to train SVC')
    # Check the score of the SVC
    log('info', 'test accuracy of SVC = ' + str(round(svc.score(X_test, y_test), 4)))
    # Check the prediction time for a single sample
    t=time()

    ###############################
    #
    # STEP 6: ADD DATASET TO THE PRECALCULATED AND WRITE TO DISK
    #
    ###############################

    myDataSet = {'color_space': color_space,
                 'spatial_size': spatial_size,
                 'hist_bins': hist_bins,
                 'orient': orient,
                 'pix_per_cell': pix_per_cell,
                 'cell_per_block': cell_per_block,
                 'hog_channel': hog_channel,
                 'spatial_feat': spatial_feat,
                 'hist_feat': hist_feat,
                 'hog_feat': hog_feat,
                 'clf': clf,
                 'X_scaler': X_scaler
                 }

    allPreTrainDataSets.append(myDataSet)
    

    # write calibration as file pkl to avoid next time calculation
    log('info', "writing trained classifier to disk " + classifierPkl)
    pickle.dump( allPreTrainDataSets, open(classifierPkl, "wb") )

    # return
    return myDataSet

def getAllFilesFromDirTree(dir):
    '''
        returns all absolute file paths from a directory tree
        dir: root directory
        return: list of all files in directory tree
    '''

    files = []
    
    for dirname, dirnames, filenames in os.walk(dir):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            pass
#            print(os.path.join(dirname, subdirname))
    
        # print path to all filenames.
        for filename in filenames:
#            print(os.path.join(dirname, filename))
            files.append(os.path.join(dirname, filename))
    
#         # Advanced usage:
#         # editing the 'dirnames' list will stop os.walk() from recursing into there.
#         if '.git' in dirnames:
#             # don't go into any .git directories.
#             dirnames.remove('.git')

    return files
