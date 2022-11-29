from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import scipy.stats
import copy


class kalman_filter():
    
    def __init__(self, x0,vx0, y0, vy0, theta0, vtheta0, dt=1) -> None:
        # initial state vector
        self.x0 = np.array([x0, vx0, y0, vy0, theta0, vtheta0])
        # state variable
        self.x = np.array([x0, vx0, y0, vy0, theta0, vtheta0])
        # posterior state variable
        self.x_post = np.array([x0, vx0, y0, vy0, theta0, vtheta0])
        self.P_post = np.zeros((6,6))
        # history of values
        self.x_upd_hist = np.zeros((6, 1))
        self.x_pred_hist = np.zeros((6, 1))
        self.P_upd_hist = np.zeros((6, 1))
        self.P_pred_hsit = np.zeros((6, 1))
        self.z_hist = np.zeros((3, 1))
        # state transition matrix 
        self.F = np.array([ [1, dt,  0,  0,  0,  0],
                            [0,  1,  0,  0,  0,  0],
                            [0,  0,  1,  dt, 0,  0],
                            [0,  0,  0,  1,  0,  0],
                            [0,  0,  0,  0,  1,  dt],
                            [0,  0,  0,  0,  0,  1]])
        # measurement matrix 
        self.H = np.array([ [1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0], 
                            [0, 0, 0, 0, 1, 0]])
        # measurement noise covariance
        self.R = np.array([ [3,  0,    0],
                            [0,  3,    0],
                            [0,   0,   3]])
        # process noise covariance 
        self.Q = 0.01 * np.array([[0.01, 0.1,    0,    0,    0,   0],
                                 [0.1,    0.25,    0,    0,    0,   0],
                                 [  0,    0, 0.01,  0.1,    0,   0],
                                 [  0,    0,  0.1,    0.25,    0,   0],
                                 [  0,    0,    0,    0, 0.01, 0.1],
                                 [  0,    0,    0,    0,  0.1,   0.25]])
        # state covariance matrix
        # since we are uncertain about the initial state,
        # we set the initial state covariance matrix to a high value
        # which is equivalent of increasing the uncertainty
        self.P = 100 * np.array([  [1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],       
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]])
        # control input matrix
        self.B = np.array([ [1, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        # control input
        self.u = np.array([0, 0, 0]).T
        # measurement
        self.z = np.array([0, 0, 0]).T
        # residual between measurement and prediction
        self.y = np.array([0, 0, 0]).T
        # kalman gain
        self.K = np.zeros((6, 3))
        pass
  

    def predict(self):
        # extrapolate state
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        # extrapolate state covariance(uncertanity?)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.x_pred_hist = np.c_[self.x_pred_hist, self.x]
        self.P_pred_hsit = np.c_[self.P_pred_hsit, self.P]
        return self.x, self.P  

    def update(self, z):
        # residual
        self.y = z - self.H.dot(self.x)
        # kalman gain
        self.K = np.dot(self.P, self.H.T).dot(np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        # state update
        self.x = self.x + self.K.dot(self.y)
        # state covariance update
        tmp = np.eye(6) - self.K.dot(self.H)
        self.P = np.dot(tmp.dot(self.P),tmp.T) + self.K.dot(self.R).dot(self.K.T)
        # save posterioir state and measurement
        self.z = copy.deepcopy(z)
        self.x_upd_hist = np.c_[self.x_upd_hist, self.x]
        self.P_upd_hist = np.c_[self.P_upd_hist, self.P]
        self.z_hist = np.c_[self.z_hist, self.z]
        

        return self.x, self.P
    
    def plot(self, y_err, x_err):
        
        plt.figure()
        plt.plot(self.x_upd_hist[0,:], self.x_upd_hist[2,:], 'r', label='state', alpha=0.5)
        plt.errorbar(self.z_hist[0,:], self.z_hist[1,:], label='measurement', yerr=y_err, xerr=x_err,  fmt='.k', alpha=0.5)
        plt.plot(self.x_pred_hist[0,:], self.x_pred_hist[2,:], 'g', label='prediction', alpha=0.5)
        plt.quiver(self.z_hist[0,:], self.z_hist[1,:], np.cos(self.z_hist[2,:]),  np.sin(self.z_hist[2,:]), alpha=0.6, width = 0.002,scale=25, headwidth=3, headlength=3, color='k')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        pass

    def plot_residuals(self):
        residuals = self.x_pred_hist - self.x_upd_hist
        plt.plot(residuals)
        plt.tight_layout()
        plt.grid()
        plt.show()

def plot_cov_ellipse(mean, cov_x, cov_y):
    s = 5.991
    return Ellipse(xy = mean, width=cov_x, height=cov_y, angle=0)
    


class sensor_position():
    def __init__(self, pos=(0,0), vel=(0,0),angle=0,vel_angle=0, noise_std = 1) -> None:
        self.pos = [pos[0], pos[1]]
        self.vel = vel
        self.angle = angle
        self.vel_angle = vel_angle
        self.noise_std = noise_std
        pass

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.angle += self.vel_angle
        
        return [self.pos[0] + randn()*self.noise_std, 
                self.pos[1] + randn()*self.noise_std,
                self.angle + randn()*self.noise_std]
    
def simulate_mouvement(N = 10, R_std = 2, init_pos = (0,0), init_vel = (1,1)):
    sensor = sensor_position(init_pos, init_vel, noise_std=R_std)
    zs = np.array([sensor.read() for _ in range(N)])
    return zs



float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
noise_std = 0.5
sensor = sensor_position(pos=(1,0),vel=(1,0))
zs = simulate_mouvement(N = 50, R_std=noise_std, init_pos=(1,0), init_vel=(1,0.2))

filter = kalman_filter(1, 1, 0, 0, 0, 0)
print("Initial state: ", filter.x)

for i, z in enumerate(zs):
    filter.predict()
    filter.update(z)
    #print("Step: ", i, "State: ", filter.x)
filter.plot(y_err=noise_std, x_err=noise_std)
#filter.plot_residuals()
