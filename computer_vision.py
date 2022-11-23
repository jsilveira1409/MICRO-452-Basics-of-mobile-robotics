import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.spatial.distance import euclidean


font = cv2.FONT_HERSHEY_SIMPLEX

#color array with blue, green and red in hsv, for the object contour coloring(in bgr)

#obstacle color boundaries(min, max) black, and the color of the contour
obst_bound = np.array([[0, 0, 0], [255,255,100], [0, 0 , 200]])

#thymio color boundaries(min,max) green and the color of the contour
robot_bound = np.array([[150, 20, 100], [175,150,255], [0, 200, 0]])

#goal color boundaries(min,max) red and the color of the contour
goal_bound = np.array([[175, 100, 100], [180,255,255], [200, 0, 0]])

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
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    video_capture.set(cv2.CAP_PROP_EXPOSURE,exposure_time)
   
    return video_capture

def image_smoothing(world, sc = 1000, ss = 1000, diameter = 30):
    smooth_world = cv2.bilateralFilter( world,
                                    d=diameter,
                                    sigmaColor = sc,
                                    sigmaSpace = ss)
    return smooth_world

def object_mask (object_to_detect, image):
    mask_bounds = object_colors[object_to_detect]
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    object_mask = cv2.inRange(image_hsv, mask_bounds[0] , mask_bounds[1])
    return object_mask

def image_segmentation(gray_image):
    segmented_world =  cv2.adaptiveThreshold(   src = gray_image, 
                                                maxValue = 255, 
                                                adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                thresholdType = cv2.THRESH_BINARY,
                                                blockSize = 21,
                                                C = 11)
    return segmented_world

def image_morph_transform(image):
    size = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))

    transformed_world  = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    transformed_world  = cv2.erode(image, kernel, iterations=3)
    return transformed_world


def object_detection(world, segmented_world, object, 
                    arc_length_precision = 0.05, min_area = 6000, max_area = 550000):
    centers = []
    areas   = []
    objects = []
    
    contours, hierarchy = cv2.findContours(segmented_world, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        

    return centers, objects, areas

def contour_treatment(world, object, centers, areas,  contours, min_dist = 250):
    color = [int(object_colors[object][2][0]), int(object_colors[object][2][1]), int(object_colors[object][2][2])]
    # remove a contour that is too close to another one
    
    unique_centers = centers
    unique_contours = contours

    #unique_centers = [p for p in centers if (euclidean(i,p) > min_dist for i in centers) ]

                
    if len(unique_centers) > 0:
        for i in range(len(unique_centers)):
            cv2.drawContours(world, [unique_contours[i]], -1, tuple(color), thickness= 3)
            cv2.circle(world, (unique_centers[i][0], unique_centers[i][1]), 3, tuple(color), -1)
            cv2.putText(world, object, (unique_centers[i][0]-10, unique_centers[i][1]-10), font, 0.5, color, 1, cv2.LINE_AA)
    return unique_centers,unique_contours

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



def computer_vision(frame, object):
    j = 0
    segmented_world = []
    centers = []
    objects = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_world = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    smooth_world = image_smoothing(hsv_world)
    mask = object_mask(object, frame)
    
    masked_world = cv2.bitwise_and(smooth_world, smooth_world, mask=mask)
    gray_world = cv2.cvtColor(masked_world, cv2.COLOR_BGR2GRAY)

    segmented_world = image_segmentation(gray_world)
    centers, objects, areas = object_detection(frame, segmented_world, object)
    unique_centers, unique_contours = contour_treatment(frame, object, centers, areas, objects)
    
    return unique_centers, unique_contours