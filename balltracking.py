#imports
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

#argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="path to the (optional) video file")
ap.add_argument("-b","--buffer",type=int,default=64,help="max buffer size")
args=vars(ap.parse_args())

#define lower and upper boundries of the "green"
#ball in the HSV color space, then initialize the
#list of tracked points
greenLower= (-10, 100, 100)
greenUpper=(10, 255, 255)
pts=deque(maxlen=args["buffer"])
# cv2.VideoCapture(0, cv2.CAP_DSHOW)


# #if a video was not supplied grab the reference to the webcam
if not args.get("video",False):
  # CAP=cv2.VideoCapture(0, cv2.CAP_DSHOW)
  # while(True):
            
  #         # Capture the video frame
  #         # by frame
  #         ret, frame = CAP.read()
        
  #         # Display the resulting frame
  #         cv2.imshow('frame', frame)
            
  #         # the 'q' button is set as the
  #         # quitting button you may use any
  #         # desired button of your choice
  #         if cv2.waitKey(1) & 0xFF == ord('q'):
  #           break
  # CAP.release()
  # cv2.destroyAllWindows()
  vs = VideoStream(src=0).start()

#otherwise grab a refernce to the video file

else:
  vs=cv2.VideoCapture(args["video"])

#allow the camera or video file to warm up
time.sleep(2.0)

#keep looping
while True:
  #grab the current frame
  frame=vs.read()

  #handle the frame from video capture or video stram
  frame=frame[1] if args.get("video,False") else frame

  #if we are viewing a video and we did not grab a frame we have rewached the end of the video
  if frame is None:
    break
  
  #resize the frame, blur it, and convert it to HSV color space
  frame=imutils.resize(frame,width=600)
  blurred=cv2.GaussianBlur(frame, (11,11), 0)
  hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

  #construct a mask for the color "green",then perform a series of dilations
  # and erosions to remove any small
  #blobs left
  mask=cv2.inRange(hsv,greenLower,greenUpper)
  mask=cv2.erode(mask,None,iterations=2)
  mask=cv2.dilate(mask,None,iterations=2)

  #find the contours in the mask and intitliaze the current
  #(x,y) center of the ball
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts=imutils.grab_contours(cnts)
  center=None

  #only proceed if atleast one contour was found
  if len(cnts)>0:
    #find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
    c=max(cnts, key=cv2.contourArea)
    ((x,y),radius)=cv2.minEnclosingCircle(c)
    M=cv2.moments(c)
    center=(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    #only proceed if the radius meets a minimum size
    if radius>10:
      #draw the circle and centroid frame
      #then update the list of tracked points
      cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
      cv2.circle(frame,center,5,(0,0,255),-1)

  #update the points queue
  pts.appendleft(center)


#drawing the contrail
  for i in range(1,len(pts)):
    #if either of the tracked points are None, ignore them
    if pts[i-1] is None or pts[i] is None:
      continue

    #otherwise compute the thickness and draw the connecting lines
    thickness=int(np.sqrt(args["buffer"]/float(i+1))*2.5)
    cv2.line(frame,pts[i-1],pts[i],(0,0,255),thickness)
    #show the frame to our screen
  cv2.imshow("Frame",frame)
  key=cv2.waitKey(1) & 0xFF

  #if q key is pressed stop the loop
  if key==ord("q"):
    break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
