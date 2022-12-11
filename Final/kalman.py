from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import time

WHEEL_DISTANCE = 95
SPEED_VAR = 40              #in mm^2/s^2        CHECK THIS
CAMERA_VAR = 0.05           #in mm^2            CHECK THIS
ROBOT_LENGTH = 60           #in mm              CHECK THIS
CAMERA_ANGLE_VAR =  0.0025  #in rad^2    CHECK THIS


def kalman_predict(previous_time, x, u, P):  
        start_time = time.time()
            
        if (previous_time != 0):
            dt = round(start_time - previous_time, 8)
        else:  
            dt = 0.2
        
        states_dim = len(x)     # x, y, theta
        control_dim = len(u)
        
        A = np.eye(states_dim)
        B = np.array([[0.5 * dt * np.cos(x[2]),     0.5 * dt * np.cos(x[2])], 
                      [0.5 * dt * np.sin(x[2]),     0.5 * dt * np.sin(x[2])],
                      [ -dt / ROBOT_LENGTH,          dt / ROBOT_LENGTH]], dtype='float')
 
        x = B.dot(u) + A.dot(x) 
        Q = SPEED_VAR * np.eye(control_dim)
        P = B.dot(Q).dot(B.T) + P

        return start_time, x, P

def kalman_update(x, z, P, sensor_available):
    R = np.diag([CAMERA_VAR, CAMERA_VAR, CAMERA_ANGLE_VAR])
    states_dim = len(x) 
    
    if sensor_available : H = np.eye(states_dim)
    else : H = np.zeros((states_dim, states_dim))

    I = z - x
    S = H.dot(P).dot(H.T)  + R
    K_gain = P.dot(H.T).dot(np.linalg.inv(S)) 
    x = x + K_gain.dot(I)
    P = P - K_gain.dot(H).dot(P)

    return x, P
    

def kalman_filter(sensor_data_available, x, u, z, P , previous_time):

    next_time, x_kal, P  = kalman_predict(previous_time, x, u, P)
    x_predicted = x_kal
    if sensor_data_available == True :
        x_kal, P = kalman_update(x, z, P, sensor_data_available)

    return next_time, x_kal, P, x_predicted


# code testing
if __name__ == "__main__":
    x = np.array([0, 0, 0], dtype=float)
    u = np.array([2, -2], dtype=float)
    P = np.eye(3) * 1
    z = np.array([0, 0, 0], dtype=float)
    next_time = 0
    for i in range(10):
        z = np.array([2*i, i, i/10], dtype=float)
        next_time, x, P, x_predicted = kalman_filter(False, x, u, z, P, next_time)
        time.sleep(1)
        print(i, np.round(x, 2), np.round(x_predicted,2))