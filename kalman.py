from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math
import time

WHEEL_DISTANCE = 95
SPEED_VAR = 0.0001              #in mm^2/s^2        CHECK THIS
CAMERA_VAR = 0.0001           #in mm^2            CHECK THIS
ROBOT_LENGTH = 60           #in mm              CHECK THIS
CAMERA_ANGLE_VAR =  0.000001        #1/(ROBOT_LENGTH*4)**2   #in rad^2    CHECK THIS


class kalman_ext_filter():
    
    def __init__(self, x0, y0, theta0, speed_l, speed_r, speed_avg = 50, dt = 0.01) -> None:
        initial_uncertainty = 1
        self.speed_variance = SPEED_VAR
        # initial state vector
        self.x0 = np.array([x0, y0, theta0])
        self.dt = dt
        self.speed_avg = speed_avg
        # state variable
        self.x = np.array([x0, y0, theta0], dtype=float)
        self.z = np.array([0,0,0],  dtype=float)
        self.x_pred = np.array([x0, y0, theta0], dtype=float)
        self.u = np.array([speed_l, speed_r], dtype=float)
        self.P = np.eye(3) * initial_uncertainty


        self.control_dim = len(self.u)
        self.states_dim = len(self.x)
        # matrix definitions
        self.A = np.eye(self.states_dim)
        self.B = np.array([ [self.speed_avg * self.dt * np.cos(self.x[2]), self.speed_avg * self.dt * np.cos(self.x[2])],
                            [self.speed_avg * self.dt * np.sin(self.x[2]), self.speed_avg * self.dt * np.sin(self.x[2])],
                            [-self.dt/WHEEL_DISTANCE                      , self.dt/WHEEL_DISTANCE] ])
                            
        self.R = np.diag([CAMERA_VAR, CAMERA_VAR, CAMERA_ANGLE_VAR])
        self.H = np.eye(self.states_dim)
        self.Q = np.eye(self.control_dim) * self.speed_variance
        self.I = np.zeros(self.states_dim)
    
        pass
  

    def predict(self, prev_time = 0):
               
        start_time = time.time()
            
        if (prev_time != 0):
            self.dt = start_time - prev_time
        
        # prediction
        self.x = self.A.dot(self.x) + self.B.dot(self.u)
        
        self.P = self.B.dot(self.Q).dot(self.B.T) + self.P

        return self.x, self.P,  start_time 

    def update(self, z):
        self.z = z
        self.I = self.z - self.x
        self.S = self.H.dot(self.P).dot(self.H.T) + self.R
        # Kalman gain
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))
        # update states
        self.x = self.x + self.K.dot(self.I)
        # update covariance
        self.P = self.P - self.K.dot(self.H).dot(self.P)

        return self.x, self.P
    
    def filter(self, z, current_time):
        self.x_pred, _, next_time = self.predict(current_time)
        self.update(z)
        if self.x[2] > np.pi:
            self.x[2] -= 2*np.pi
        elif self.x[2] < -np.pi:
            self.x[2] += 2*np.pi

        return self.x, self.P, next_time

#current_time = 0
#x_pred = [0,0,0]
#time_step = 1
#z = [[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0]]
#z2 = [[10*i , 5*i,0] for i in range(1000)]
#
#filter = kalman_ext_filter(0,0,0,1,0.5,10,1)
#
#for measurement in z2:    
#    x_pred, P_pred, current_time =  filter.filter(measurement, current_time)
#    print(x_pred)
#