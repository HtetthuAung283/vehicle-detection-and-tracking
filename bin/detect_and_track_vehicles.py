#!/usr/bin/python

"""
    Project 5
    Udacity Self-Driving-Car-Engineer Nanodegree

    Vehicle Detection And Tracking
    
    3) Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
    4) Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
    * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
    * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
    * Estimate a bounding box for vehicles detected.
"""    

import os
import sys
import argparse
from time import time
from moviepy.editor import VideoFileClip

# add lib to path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../lib")
from helper_vehicle_detection import *
from detection import Detection

# setting etc dir
etcDir = os.path.dirname(os.path.realpath(__file__))+'/../etc'

version = "0.1"
date = "2017-05-16"

# Definieren der Kommandozeilenparameter
parser = argparse.ArgumentParser(description='a tool for detecting lane lines in images and videos',
                                 epilog='author: alexander.vogel@prozesskraft.de | version: ' + version + ' | date: ' + date)
parser.add_argument('--image', metavar='PATH', type=str, required=False,
                   help='image from a front facing camera. to detect lane lines')
parser.add_argument('--video', metavar='PATH', type=str, required=False,
                   help='video from a front facing camera. to detect lane lines')
parser.add_argument('--startTime', metavar='INT', type=int, required=False,
                   help='when developing the image pipeline it can be helpful to focus on the difficult parts of an video. Use this argument to shift the entry point. Eg. --startTime=25 starts the processing pipeline at the 25th second after video begin.')
parser.add_argument('--endTime', metavar='INT', type=int, required=False,
                   help='Use this argument to shift the exit point. Eg. --endTime=30 ends the processing pipeline at the 30th second after video begin.')
parser.add_argument('--unroll', action='store_true',
                   help='Use this argument to unroll the resulting video in single frames.')
parser.add_argument('--visLog', metavar='INT', type=int, action='store', default=False,
                   help='for debugging or documentation of the pipeline you can output the image at a certain processing step \
                   1=detections, \
                   2=heatmap, \
                   3=thresholded_heatmap \
                   4=result'
                   )
parser.add_argument('--format', metavar='STRING', type=str, action='store', default='normal',
                   help='to visualize several steps of the image pipeline and plot them in one single image. use --format=collage4 for a 4-image-collage')
parser.add_argument('--outDir', metavar='PATH', action='store', default='output_directory_'+str(time()),
                   help='directory for output data. must not exist at call time. default is --outDir=output_directory_<time>')
parser.add_argument('--mlDir', metavar='PATH', action='store', required=False, default=etcDir + '/ml_train_img',
                   help='directory for machine learning training images. directory must contain 2 subdirectories "vehicles" and "non-vehicles". default is --calDir=etc/ml_train_img')

args = parser.parse_args()

map_int_name = {    
                    0: '00_original',
                    1: '01_detections',
                    2: '02_heatmap',
                    3: '03_thresh_heatmap', 
                    4: '99_result',
                    False: '99_result',
                }



errors = 0

# check whether image or video was supplied
if not args.image and not args.video:
    log('error', 'you need to provide at least one image or video. try --help for help.')
    errors += 1

# check if all provided images exist
if args.image:
    if not os.path.isfile(args.image):
        log('error', 'image does not exist:'+ args.image)
        errors += 1

# check if all provided videos exist
if args.video:
    if not os.path.isfile(args.video):
        log('error', 'video does not exist:'+ args.video)
        errors += 1
        
# check if mlDir does NOT exist
if not os.path.isdir(args.mlDir):
    log('error', 'directory with images for machine learning does not exist: ' + args.mlDir)
    errors += 1

# check if outDir does exist
if os.path.isdir(args.outDir):
    log('error', 'output directory already exists. please delete or rename:' + args.outDir)
    errors += 1

# check if format is normal|collage4|collage9
if args.format != 'normal' and args.format != 'collage4':
    log('error', '--format=' + args.format + ' does not exist. try --help for help')
    errors += 1

if errors > 0:
    log('fatal', str(errors) + ' error(s) occured. please correct and try again.')
    sys.exit(1)
    
log('info', '--outDir='+args.outDir)
log('info', '--mlDir='+args.mlDir)

log('info', '--visLog=' + str(args.visLog))
log('info', '--format=' + args.format)

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
#======================
#
# CREATE A SVM CLASSIFIER WITH TRAINING DATA
#
#----------------------
dict_classifier = createClassifier(args.mlDir, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)


#======================
#
# DETECT VEHICLES
#
#----------------------

detection = Detection()
frameNr = 1

def process_image(img, detection=detection):
    result, detection = detect_vehicles(img, detection, dict_classifier['clf'], dict_classifier['X_scaler'], color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, x_start_stop=[None, None], y_start_stop=[380, 650], retNr=args.visLog, format=args.format)
    
    if args.unroll:
        global frameNr
        writeImage(result, args.outDir, 'frame_'+str(frameNr))
        frameNr += 1
    
    return result



if args.image:
    
    # read image
    img = mpimg.imread(args.image)
    result, detection = detect_vehicles(img, detection, dict_classifier['clf'], dict_classifier['X_scaler'], color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, x_start_stop=[None, None], y_start_stop=[380, 650], subsampling=False, retNr=args.visLog, format=args.format)
    
    print(map_int_name[args.visLog])
    writeImage(result, args.outDir, map_int_name[args.visLog], cmap=None)

if args.video:
    video_output = args.outDir + '/video_out.mp4'
    clip = VideoFileClip(args.video, audio=False)

    if not os.path.isdir(args.outDir):
        log('info', 'creating output directory: ' + args.outDir)
        os.mkdir(args.outDir)

    subclip = None
    if args.startTime and args.endTime:
        log('info', '--startTime='+str(args.startTime))
        log('info', '--endTime='+str(args.endTime))
        subclip = clip.subclip(args.startTime, args.endTime)
    elif args.startTime:
        log('info', '--startTime='+str(args.startTime))
        log('info', '--endTime is set to end of video.')
        subclip = clip.subclip(t_start=args.startTime)
    elif args.endTime:
        log('info', '--startTime is set to start of video')
        log('info', '--endTime='+str(args.endTime))
        subclip = clip.subclip(t_end=args.endTime)
    else:
        log('info', '--startTime is set to start of video')
        log('info', '--endTime is set to end of video.')
        subclip = clip

    white_clip = subclip.fl_image(process_image) #NOTE: this function expects color images!!
    

    white_clip.write_videofile(video_output, audio=False)
    

