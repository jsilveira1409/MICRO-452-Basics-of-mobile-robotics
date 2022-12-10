import numpy as np
import math
import time
from computer_vision import *
from dijkstra import compute_shortest_path
from kalman import *
import tdmclient.notebook



# parametres
MIN_DIST = 10
ANGLE_TOLERANCE = 0.30
PERIOD = 0.25
SPEED_AVG = 300
ROBOT_SPEED_TO_MM = 140/500



def get_angle_between(vec1, vec2):
    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec2 / np.linalg.norm(vec2)

    return np.arccos(np.dot(vec1_unit, vec2_unit))

def distance (x1, y1, x2, y2):
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

def wrap_angle(angle):
    if angle > math.pi:
        angle = angle - 2*math.pi
    elif angle < -math.pi:
        angle = angle + 2*math.pi
    return angle

def controller(angle):
    kp_rot = 70

    if abs(angle) > ANGLE_TOLERANCE:
        speed_l =  - kp_rot*(angle)
        speed_r =  + kp_rot*(angle)

    else:
        speed_l = SPEED_AVG
        speed_r = SPEED_AVG
    return int(speed_l), int(speed_r)

