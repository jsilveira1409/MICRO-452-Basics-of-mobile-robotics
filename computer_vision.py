import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import copy
from scipy.spatial.distance import euclidean

CAMERA_WIDTH = 1720
CAMERA_HEIGHT = 960
PIXEL_TO_MM = 0.73826
font = cv2.FONT_HERSHEY_SIMPLEX

#color array with blue, green and red in hsv, for the object contour coloring(in bgr)

#obstacle color boundaries(min, max) black, and the color of the contour
obst_bound = np.array([[0, 0, 0], [180,150,80], [0, 0 , 200]])

#thymio color boundaries(min,max) yellow and the color of the contour
robot_bound = np.array([[80, 200, 100], [120, 255,255], [0, 200, 0]])

#goal color boundaries(min,max) red and the color of the contour
goal_bound = np.array([[140, 100, 100], [210, 255,255], [200, 0, 255]])

#ruler color boundaries(min,max) green and the color of the contour
reference_bound = np.array([[45, 50, 100], [60, 255,255], [200, 255, 0]])

object_colors =   {'obstacle'       : obst_bound, 
                    'robot'         : robot_bound, 
                    'goal'          : goal_bound,
                    'reference'          : reference_bound
                    }

# pixel per cm ratio
ratio = 0


def setup_camera(video_capture, exposure_time = None):
    #cv2.namedWindow("Computer Vision", cv2.WINDOW_NORMAL)
    #video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH )
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if exposure_time != None:
        video_capture.set(cv2.CAP_PROP_EXPOSURE, exposure_time)
    else:
        video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,3)   
    

def read_camera(video_capture, exp = None):
    vid = setup_camera(video_capture, exposure_time = exp)
    
    while(True):
        ret, frame = vid.read()
        if ret == False:
            print("Cannot read frame")
            exit()
        cv2.imshow("frame",frame)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    vid.release()
    # Closes all the frames
    cv2.destroyAllWindows()



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


def object_detection(object, img, img_masked, show_image = False, arc_length_precision = 0.05, min_area = 7000, max_area = 400000):
    centers = []
    areas   = []
    objects = []
    color = [int(object_colors[object][2][0]), int(object_colors[object][2][1]), int(object_colors[object][2][2])]
    
    contours, _ = cv2.findContours(img_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # approximate contour
        epsilon = arc_length_precision * cv2.arcLength(curve = contour, closed = True)
        contour_approximation = cv2.approxPolyDP(contour, epsilon,True)
        # calculate centroid of each contour approximation
        M = cv2.moments(contour_approximation)
        # moment is zero for open contours (or contours with no area)
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


    if show_image:
        if len(centers) > 0:
            for i in range(len(centers)):
                cv2.drawContours(img, [objects[i]], -1, tuple(color), thickness= 3)
                cv2.circle(img, (centers[i][0], centers[i][1]), 3, tuple(color), -1)
                cv2.putText(img, object, (centers[i][0]-10, centers[i][1]-10), font, 0.5, color, 1, cv2.LINE_AA)
        
    return centers, objects, areas

def computer_vision(img=None, object = None, show_image = False):
    global video_capture
    if img is None:
        ret, img = video_capture.read()

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
    # TODO: not sure about passing the original frame to it, as it would be used to process the other objects
    centers, contours, areas = object_detection(object, img, mask, show_image)

    return centers, contours, img_processed



# TODO: redefine robot_contour to be 1D
def get_robot_position(frame, robot_center, robot_contour):
    center = np.array(robot_center, dtype="object")
    contour = np.array(robot_contour, dtype="object")
    center = np.reshape(np.ravel(center), (-1,2))
    contour = np.reshape(np.ravel(contour), (-1,2)) 
    
    min_dist = 0
    max_index = 0
    dir_vector = alpha = 0
    for i in range(len(contour)):
        if euclidean(contour[i], center[0]) > min_dist:
            min_dist = euclidean(contour[i], center[0])
            max_index = i
    dir_vector = contour[max_index] - center[0]
    
    alpha = np.arctan2(dir_vector[1], dir_vector[0])
    cv2.arrowedLine(frame, center[0], contour[max_index], (0, 0, 255), 2)

    return dir_vector, alpha

def format_contour(contours):
    c = []
    for contour in contours:
        L = contour.ravel()
        it = iter(L)
        c.append(list(zip(it, it)))
    return c


def cv_start(video_capture, exposure = None, show_image = False, nb_tries = 5):
    global ratio
    setup_camera(video_capture, exposure)
    # read first 100 frames, to give time to the camera to adapt to the light
    video_capture.read(300)
    robot_center = []
    # read frame for further analysis
    ret, frame = video_capture.read()
    robot_pos = []
    cnt = 0
    cv_success = False
    ratio = 0
    
    while(True):
        # detect each type of object
        if ret == True:
            robot_center, robot_contour, _ = computer_vision(frame, 'robot', show_image)
            _, obst_contours, _ = computer_vision(frame, 'obstacle', show_image)
            goal_center, goal_contours, _ = computer_vision(frame, 'goal', show_image)
            #ref_center, ref_contour, _ = computer_vision(frame, 'reference', show_image)        
            
        #if len(robot_center) == 1 and len(goal_center) == 1 and len(ref_center) == 1:
        if len(robot_center) == 1 and len(goal_center) == 1:
            # the reference object is detected, so we can calculate the ratio
            # it is a 7.5cm square object
            #ref_contour = format_contour(ref_contour)
            #for f in ref_contour:
            #    ratio += 75 / euclidean(f[0], f[1]) 
#
            #ratio = ratio / len(ref_contour)
            #print("ratio = ", ratio)
#
            cv_success = True
            break
        else:
            ret, frame = video_capture.read()
            cnt = cnt + 1
            print(cnt)
            if cnt > nb_tries:
                print("Either the robot or the goal is not visible/detectable")
                break


    if cv_success:
        obst_contours = np.array(obst_contours, dtype=object)
        
        # get robot direction
        _, alpha = get_robot_position(frame, robot_center, robot_contour)
        robot_pos = [robot_center[0][0], robot_center[0][1], alpha]
            
    #return cv_success, obst_contours, robot_pos, goal_center, frame
    
    return cv_success, obst_contours, robot_pos, goal_center[0], frame
    
def pixel_to_metric(px_point):
    metric_point = np.array(px_point) * PIXEL_TO_MM
    return metric_point

def metric_to_pixel(metric_point):
    px_point = np.array(metric_point) / PIXEL_TO_MM
    return px_point

def draw_path(frame, path):
    for i in range(len(path)-1):
        cv2.arrowedLine(frame, path[i], path[i+1], (200, 0, 0), 5)
    return frame

def invert_coordinates(point):
    return [point[0], CAMERA_HEIGHT - point[1]]
def revert_coordinates(point):
    return [point[0], CAMERA_HEIGHT + point[1]]




    #if show_image:
    #    # show the frames
    #    cv2.imshow('Computer Vision', frame)
    #    while True:
    #        if cv2.waitKey(1) & 0xFF == ord('q'):
    #            break
    #    
    #    # Release the capture
    #    video_capture.release() 
    #    cv2.destroyAllWindows()
        

#
##read image as rgb
#img = cv2.imread('test.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#fig, ax = plt.subplots(2, 3, figsize=(20, 10))
#
#img_processed = img.copy()
#img_hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
#img_smooth = image_smoothing(img_hsv)
#mask = object_mask('robot', img_smooth)
#centers, contours, areas = object_detection('robot', img_processed, mask)
#
##plot images
#ax[0,0].imshow(img)
#ax[0,0].set_title('Original Image')
#ax[0,1].imshow(img_hsv)
#ax[0,1].set_title('HSV Image')
#ax[0,2].imshow(img_smooth)
#ax[0,2].set_title('Smoothed Image')
#ax[1,0].imshow(mask)
#ax[1,0].set_title('Mask')

#ax[1,1].imshow(img_processed)
#ax[1,1].set_title('Segmented Image')
#ax[1,2].imshow(img_processed)
#ax[1,2].set_title('Segmented Image')
#
#fig.tight_layout()
#fig.savefig('data1.jpeg')
#