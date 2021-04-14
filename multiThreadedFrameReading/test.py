import cv2 #version 4.2.0
import imutils
import argparse
from uitls import threadedStream
import time

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=100,
            help="# of frames to loop over FPS test")
    ap.add_argument("-d", "--display", type=int, default=-1,
            help="whether or not frames should be displayed")
    args = vars(ap.parse_args())

    # grab a pointer to the video stream and initialize the FPS counter
    print ("[INFO] sampling frames from webcam")
    stream = threadStream(src=0).start()
    
    fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
    print("FPS: {0}".format(fps))
    
    numFrameToCapture = 1000
    print("Capturing {0} frames".format(numFrameToCapture))
    start = time.time()
    
    for i in range(0, numFrameToCapture):
        ret, frame = video.read()
    end = time.time()
    sec = end - start
    
    fps  = numFrameToCapture / sec
    
    print("Estimated frames per second : {0}".format(fps))
    stream.release()
