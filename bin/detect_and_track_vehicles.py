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
parser.add_argument('--visLog', metavar='INT', type=int, action='store', default=False,
                   help='for debugging or documentation of the pipeline you can output the image at a certain processing step \
                   1=undistorted image \
                   2=grayscale \
                   3=binary mask b of lab \
                   4=binary mask l of luv \
                   5=combination of binary masks \
                   6=unwarped binary with polygon \
                   7=warped binary with polygon \
                   8=warped binary \
                   9=histogram \
                   10=detected lines \
                   11=undistorted with detected lines \
                   12=result with text'
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
                    2: '01_hog',
                    3: '02_color_hist',
                    4: '03_heatmap',
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
# CREATE A SVM CLASSIFIER WITH TRAINING DATA
#
#----------------------
clf = createClassifier(args.mlDir)


#======================
#
# DETECT VEHICLES
#
#----------------------

detection = Detection()

def process_image(img, detection=detection):
    result, detection = detectVehicles(img, clf, args.outDir, args.visLog, detection, args.format, sobel_kernel=9, mag_sobelxy_thresh=(70, 100), hls_thresh=(120, 255), lab_thresh=(160, 255), luv_thresh=(200, 255))
    return result



if args.image:
    
    # read image
    img = mpimg.imread(args.image)
    result, detection = detectVehicles(img, mtx, dist, args.outDir, args.visLog, leftLine, rightLine, args.format, sobel_kernel=9, mag_sobelxy_thresh=(70, 100), hls_thresh=(120, 255), lab_thresh=(160, 255), luv_thresh=(200, 255))
    
    print(map_int_name[args.visLog])
    writeImage(result, args.outDir, map_int_name[args.visLog], cmap=None)

if args.video:
    video_output = args.outDir + '/video_out.mp4'
    clip = VideoFileClip(args.video, audio=False)

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
    
    if not os.path.isdir(args.outDir):
        log('info', 'creating output directory: ' + args.outDir)
        os.mkdir(args.outDir)

    white_clip.write_videofile(video_output, audio=False)
    

