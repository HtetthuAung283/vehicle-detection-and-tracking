#!/usr/bin/python

"""
    unrolls a video in separate frames
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
date = "2017-05-18"

# Definieren der Kommandozeilenparameter
parser = argparse.ArgumentParser(description='a tool for unrolling a video in single images',
                                 epilog='author: alexander.vogel@prozesskraft.de | version: ' + version + ' | date: ' + date)
parser.add_argument('--video', metavar='PATH', type=str, required=False,
                   help='video from a front facing camera. to detect lane lines')
parser.add_argument('--startTime', metavar='INT', type=int, required=False,
                   help='when developing the image pipeline it can be helpful to focus on the difficult parts of an video. Use this argument to shift the entry point. Eg. --startTime=25 starts the processing pipeline at the 25th second after video begin.')
parser.add_argument('--endTime', metavar='INT', type=int, required=False,
                   help='Use this argument to shift the exit point. Eg. --endTime=30 ends the processing pipeline at the 30th second after video begin.')
parser.add_argument('--outDir', metavar='PATH', action='store', default='output_directory_'+str(time()),
                   help='directory for output data. must not exist at call time. default is --outDir=output_directory_<time>')

args = parser.parse_args()

errors = 0

# check if all provided videos exist
if args.video:
    if not os.path.isfile(args.video):
        log('error', 'video does not exist:'+ args.video)
        errors += 1
        
# check if outDir does exist
if os.path.isdir(args.outDir):
    log('error', 'output directory already exists. please delete or rename:' + args.outDir)
    errors += 1

if errors > 0:
    log('fatal', str(errors) + ' error(s) occured. please correct and try again.')
    sys.exit(1)

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
    
#    plt.clf()

frameNr = 1;

def process_image(img):
    
    global frameNr
    writeImage(img, args.outDir, 'frame_'+str(frameNr))
    frameNr += 1
    return img




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
    

