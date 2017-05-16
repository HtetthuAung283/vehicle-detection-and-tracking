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
    '''
    
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
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

    if format == 'collage4':
        return genCollage(4, imageBank), leftLine, rightLine
    elif format == 'collage9':
        return genCollage(9, imageBank), leftLine, rightLine
    
    
    return resultImage, leftLine, rightLine

def genCollage(amount, imageBank):
    '''
        generating a 2x2 or 3x3 collage
        amount: 4 -> 2x2 collage; 9 _> 3x3 collage
        return: imageCollage
    '''
    resultImage = None
    

    if amount == 4:
        row1 = cv2.hconcat((imageBank[1], imageBank[5]))
        row2 = cv2.hconcat((imageBank[10], imageBank[12]))
        resultImage = cv2.vconcat((row1, row2))
        resultImage = cv2.resize(resultImage, (1920, int((1920/resultImage.shape[1]) * resultImage.shape[0])))
    
    elif amount == 9:
        row1 = cv2.hconcat((imageBank[1], imageBank[2], imageBank[4]))
        row2 = cv2.hconcat((imageBank[5], imageBank[6], imageBank[8]))
        row3 = cv2.hconcat((imageBank[9], imageBank[10], imageBank[12]))
        resultImage = cv2.vconcat((row1, row2, row3))
        resultImage = cv2.resize(resultImage, (1920, int((1920/resultImage.shape[1]) * resultImage.shape[0])))
    
    return resultImage

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
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



def createClassifier(mlDir):
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
    
    # if one exists, then the classifier can be loaded from there instead of running the whole new training process
    if os.path.isfile(classifierPkl):
        log('info', 'precalculated classifier file found - loading that: ' + classifierPkl)
        clf = pickle.load(open(classifierPkl, "rb"))
        return clf
    
    ###############################
    #
    # STEP 2: DETERMINING ALL FILEPATHS OF VEHICLE- AND NON-VEHICLE IMAGES
    #
    ###############################
    imagesVehicles = getAllFilesFromDirTree(mlDir + '/vehicles')
    imagesNonVehicles = getAllFilesFromDirTree(mlDir + '/non-vehicles')
    log('info', 'determining paths for learning images. ' + str(len(imagesVehicles) + len(imagesNonVehicles)) + ' files found.')
    
    ###############################
    #
    # STEP 3: READ IMAGE, GENERATE FEATURES AND LABELS
    #
    ###############################
    log('info', 'reading ' + str(len(imagesVehicles)) + ' images of vehicles and converting them into feature vectors.')

    featureVectorsVehicles = extract_features(imagesVehicles, color_space='LUV', spatial_size=(32, 32),
                                            hist_bins=32, orient=9,
                                            pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                                            spatial_feat=False, hist_feat=False, hog_feat=True)
    
    log('info', 'reading ' + str(len(imagesNonVehicles)) + ' images of non-vehicles and converting them into feature vectors.')

    featureVectorsNonVehicles = extract_features(imagesNonVehicles, color_space='LUV', spatial_size=(32, 32),
                                            hist_bins=32, orient=9,
                                            pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                                            spatial_feat=False, hist_feat=False, hog_feat=True)
    
    

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



    # write calibration as file pkl to avoid next time calculation
    log('info', "writing trained classifier to disk " + classifierPkl)
    pickle.dump( clf, open(classifierPkl, "wb") )


    # return
    return clf

def getAllFilesFromDirTree(dir):
    '''
        returns all absolute file paths from a directory tree
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
