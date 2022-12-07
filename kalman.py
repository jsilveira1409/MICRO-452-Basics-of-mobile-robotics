from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import scipy.stats
import copy
import time

WHEEL_DISTANCE = 95
SPEED_VAR = 0.1   # check this
CAMERA_VAR = 0.1  # check this
CAMERA_ANGLE_VAR = 0.1  # check this

class kalman_ext_filter():
    
    def __init__(self, x0, y0, theta0, speed_l, speed_r, ratio, speed_avg = 50,dt = 0.01) -> None:
        initial_uncertainty = 1500
        self.speed_variance = SPEED_VAR
        self.convert_px_to_mm = ratio
        # initial state vector
        self.x0 = np.array([x0, y0, theta0])
        self.dt = dt
        self.speed_avg = speed_avg
        # state variable
        self.x = np.array([x0, y0, theta0], dtype=float)
        self.x_pred = np.array([x0, y0, theta0], dtype=float)
        self.u = np.array([speed_l, speed_r], dtype=float)
        self.P = np.eye(3) * initial_uncertainty

        self.control_dim = len(self.u)
        self.states_dim = len(self.x)
        # matrix definitions
        self.A = np.eye(self.states_dim)
        self.B = np.array([ [self.speed_avg * self.dt * np.cos(self.x[2]), self.speed_avg * self.dt * np.cos(self.x[2])],
                            [self.speed_avg * self.dt * np.sin(self.x[2]), self.speed_avg * self.dt * np.sin(self.x[2])],
                            [self.dt/WHEEL_DISTANCE                      ,-self.dt/WHEEL_DISTANCE] ])
        self.R = np.diag([CAMERA_VAR, CAMERA_VAR, CAMERA_ANGLE_VAR])
        self.H = np.eye(self.states_dim)
        self.Q = np.eye(self.control_dim) * self.speed_variance
        self.I = np.zeros(self.states_dim,1)
        

        pass
  

    def predict(self):
        # convert to metrics
        self.x[0] = self.x[0] * self.convert_px_to_mm
        self.x[1] = self.x[1] * self.convert_px_to_mm
        # prediction
        self.x = self.A.dot(self.x) + self.B.dot(self.u)
        
        self.P = self.B.dot(self.Q).dot(self.B.T) + self.P

        return self.x, self.P  

    def update(self, z):
        self.I = self.z - self.x
        self.S = self.H.dot(self.P).dot(self.H.T) + self.R
        # Kalman gain
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))
        # update states
        self.x = self.x + self.K.dot(self.I)
        # update covariance
        self.P = self.P - self.K.dot(self.H).dot(self.P)

        return self.x, self.P
    
    def filter(self, z):
        self.x_pred, _ = self.predict()
        self.update(z)
        
        return self.x, self.P