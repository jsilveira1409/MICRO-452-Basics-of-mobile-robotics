import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.spatial.distance import euclidean


font = cv2.FONT_HERSHEY_SIMPLEX

#color array with blue, green and red in hsv, for the object contour coloring(in bgr)

#obstacle color boundaries(min, max) black, and the color of the contour
obst_bound = np.array([[0, 0, 0], [180,255,60], [0, 0 , 200]])

#thymio color boundaries(min,max) green and the color of the contour
robot_bound = np.array([[60, 100, 60], [135, 120,255], [0, 200, 0]])

#goal color boundaries(min,max) red and the color of the contour
goal_bound = np.array([[150, 60, 10], [200, 255,255], [200, 0, 255]])


object_colors =   {'obstacle'       : obst_bound, 
                    'robot'         : robot_bound, 
                    'goal'          : goal_bound
                }


def setup_camera(exposure_time):
    #cv2.namedWindow("Computer Vision", cv2.WINDOW_NORMAL)

    video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 )
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_EXPOSURE,exposure_time)
   
    return video_capture

def image_smoothing(world, sc = 1000, ss = 3000, diameter = 20):
    smooth_world = cv2.bilateralFilter( world, d=diameter, sigmaColor = sc, sigmaSpace = ss)
    #smooth_world = cv2.medianBlur(world, 5)
    return smooth_world

def object_mask (object_to_detect, image_hsv):
    mask_bounds = object_colors[object_to_detect]
    min_bound = np.array(mask_bounds[0], np.uint8)
    max_bound = np.array(mask_bounds[1], np.uint8)
    object_mask = cv2.inRange(image_hsv, min_bound, max_bound)
    return object_mask

def image_segmentation(im_gray):
    #segmented_world =  cv2.adaptiveThreshold(   src = gray_image, maxValue = 200, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType = cv2.THRESH_BINARY,blockSize = 27,C = 11)
    ret, im_segmented = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY)
    return im_segmented

def image_morph_transform(image):
    size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))

    transformed_world  = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    transformed_world  = cv2.erode(image, kernel, iterations=3)
    return transformed_world


def object_detection(object, img, img_masked, arc_length_precision = 0.05, min_area = 3000, max_area = 400000):
    centers = []
    areas   = []
    objects = []
    color = [int(object_colors[object][2][0]), int(object_colors[object][2][1]), int(object_colors[object][2][2])]
    
    contours, _ = cv2.findContours(img_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        #approximate contour
        epsilon = arc_length_precision * cv2.arcLength(curve = contour, closed = True)
        contour_approximation = cv2.approxPolyDP(contour, epsilon,True)
        #calculate centroid of each contour approximation
        M = cv2.moments(contour_approximation)
        #moment is zero for open contours (or contours with no area)
        if M['m00'] == 0:       
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    
        #calculate area
        area = cv2.contourArea(contour)
        
        if area <= min_area or area >= max_area:
            continue
        centers.append([cx, cy])
        areas.append(area)
        objects.append(contour_approximation)



    if len(centers) > 0:
        for i in range(len(centers)):
            cv2.drawContours(img, [objects[i]], -1, tuple(color), thickness= 3)
            cv2.circle(img, (centers[i][0], centers[i][1]), 3, tuple(color), -1)
            cv2.putText(img, object, (centers[i][0]-10, centers[i][1]-10), font, 0.5, color, 1, cv2.LINE_AA)
    
    return centers, objects, areas


# TODO: redefine robot_contour to be 1D
def get_robot_position(robot_center, robot_contour):
    min_dist = 0
    min_index = 0
    for i in range(len(robot_contour)):
        if euclidean(robot_contour[0][i], robot_center) > min_dist:
            min_dist = euclidean(robot_contour[i], robot_center)
            min_index = i
    dir_vector = robot_contour[0][min_index] - robot_center
    alpha = np.arctan2(dir_vector[1], dir_vector[0])
    return robot_center[0], robot_center[1], alpha


def computer_vision(img, object):
    img_processed = img.copy()
    # 1. convert to hsv color space
    # it is easier to filter colors in the HSV color-space.
    img_hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
    # 2. smooth the image
    # this will remove any small white noises
    img_smooth = image_smoothing(img_hsv)
    # 3. create mask and mask the image         
    # create a mask of the color we are looking for
    mask = object_mask(object, img_smooth)
    #img_masked = cv2.bitwise_and(img_smooth, img_smooth, mask=mask)
    #img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    # 4. segment the image
    #   NB : the im_masked is already segmented, so no real need for this part.
    #im_segmented = image_segmentation(im_masked_gray)

    # 5. detect objects
    centers, contours, areas = object_detection(object, img_processed, mask)

    return centers, contours, img_processed


#read image as rgb
img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
fig, ax = plt.subplots(2, 3, figsize=(20, 10))

img_processed = img.copy()
img_hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
img_smooth = image_smoothing(img_hsv)
mask = object_mask('robot', img_smooth)
centers, contours, areas = object_detection('robot', img_processed, mask)

#plot images
ax[0,0].imshow(img)
ax[0,0].set_title('Original Image')
ax[0,1].imshow(img_hsv)
ax[0,1].set_title('HSV Image')
ax[0,2].imshow(img_smooth)
ax[0,2].set_title('Smoothed Image')
ax[1,0].imshow(mask)
ax[1,0].set_title('Mask')
ax[1,1].imshow(img_processed)
ax[1,1].set_title('Segmented Image')
ax[1,2].imshow(img_processed)
ax[1,2].set_title('Segmented Image')

fig.tight_layout()
fig.savefig('data1.jpeg')
