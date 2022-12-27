from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math
import time
#
WHEEL_DISTANCE = 95
SPEED_VAR = 20              #in mm^2/s^2        CHECK THIS
CAMERA_VAR = 0.01           #in mm^2            CHECK THIS
ROBOT_LENGTH = 60           #in mm              CHECK THIS
CAMERA_ANGLE_VAR =  0.01  #in rad^2    CHECK THIS
#
#
#class kalman_ext_filter():
 #   
 #   def __init__(self, x0, y0, theta0, speed_l, speed_r, dt = 0.2) -> None:
 #       initial_uncertainty = 5000
 #       self.speed_variance = SPEED_VAR
 #       # initial state vector
 #       self.x0 = np.array([x0, y0, theta0])
 #       self.dt = dt
 #       # state variable
 #       self.x = np.array([x0, y0, theta0], dtype=float)
 #       self.z = np.array([0,0,0],  dtype=float)
 #       self.x_pred = np.array([0,0,0], dtype=float)
 #       self.u = np.array([speed_l, speed_r], dtype=float)
 #       self.P = np.eye(3) * initial_uncertainty
#
#
 #       self.control_dim = len(self.u)
 #       self.states_dim = len(self.x)
 #       # matrix definitions
 #       self.A = np.eye(self.states_dim)
 #       self.B = np.array([ [ 0.5 * self.dt * np.cos(self.x[2]), 0.5 * self.dt * np.cos(self.x[2])],        # porque tem que multiplicar por 0.5?
 #                           [ 0.5 * self.dt * np.sin(self.x[2]), 0.5 * self.dt * np.sin(self.x[2])],        # pra fazer uma media entre velocidade direita esquerda?    
 #                           [-self.dt/WHEEL_DISTANCE                      , self.dt/WHEEL_DISTANCE] ])
 #                           
 #       self.R = np.diag([CAMERA_VAR, CAMERA_VAR, CAMERA_ANGLE_VAR])
 #       self.H = np.eye(self.states_dim)
 #       self.Q = np.eye(self.control_dim) * self.speed_variance
 #       self.I = np.zeros(self.states_dim)
 #   
 #       pass
 # 
#
 #   def predict(self, prev_time = 0):
 #              
 #       start_time = time.time()
 #           
 #       if (prev_time != 0):
 #           self.dt = round(start_time - prev_time, 4)
 #           
 #       
 #       # prediction
 #       self.x = self.A.dot(self.x) + self.B.dot(self.u)
 #       
 #       self.P = self.B.dot(self.Q).dot(self.B.T) + self.P
#
 #       return self.x, self.P,  start_time 
#
 #   def update(self, z):
 #       
 #       self.I = self.z - self.x
 #       self.S = self.H.dot(self.P).dot(self.H.T) + self.R
 #       # Kalman gain
 #       self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))
 #       # update states
 #       self.x = self.x + self.K.dot(self.I)
 #       # update covariance
 #       self.P = self.P - self.K.dot(self.H).dot(self.P)
 #       
 #       self.z = z
 #       return self.x, self.P
 #   
 #   def filter(self, z, current_time):
 #       if self.x[2] > np.pi:
 #           self.x[2] -= 2*np.pi
 #       elif self.x[2] < -np.pi:
 #           self.x[2] += 2*np.pi
#
 #       self.x_pred, _, next_time = self.predict(current_time)
#
 #       
 #       self.x, self.P = self.update(z)
 #       return self.x, self.P, self.x_pred, next_time
#
#
#
## function version

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