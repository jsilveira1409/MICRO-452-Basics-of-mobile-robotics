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
        


        return self.x, self.P
    
    def plot(self, y_err, x_err, fig=None, ax=None):
        if fig is None or ax is None:
            plt.plot(self.x_upd_hist[0,:], self.x_upd_hist[2,:], 'r', label='state', alpha=0.5)
            plt.errorbar(self.z_hist[0,:], self.z_hist[1,:], label='measurement', yerr=y_err, xerr=x_err,  fmt='.k', alpha=0.5)
            plt.plot(self.x_pred_hist[0,:], self.x_pred_hist[2,:], 'g', label='prediction', alpha=0.5)
            plt.quiver(self.z_hist[0,:], self.z_hist[1,:], np.cos(self.z_hist[2,:]),  np.sin(self.z_hist[2,:]), alpha=0.6, width = 0.002,scale=25, headwidth=3, headlength=3, color='k')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.pause(0.05)
        else:
            ax.plot(self.x_upd_hist[0,:], self.x_upd_hist[2,:], 'r', label='state', alpha=0.5)
            ax.errorbar(self.z_hist[0,:], self.z_hist[1,:], label='measurement', yerr=y_err, xerr=x_err,  fmt='.k', alpha=0.5)
            ax.plot(self.x_pred_hist[0,:], self.x_pred_hist[2,:], 'g', label='prediction', alpha=0.5)
            ax.quiver(self.z_hist[0,:], self.z_hist[1,:], np.cos(self.z_hist[2,:]),  np.sin(self.z_hist[2,:]), alpha=0.6, width = 0.002,scale=25, headwidth=3, headlength=3, color='k')
            ax.legend()
            ax.grid()
            fig.tight_layout()
            plt.pause(0.05)


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



#float_formatter = "{:.2f}".format
#np.set_printoptions(formatter={'float_kind':float_formatter})
#noise_std = 0.5
#sensor = sensor_position(pos=(1,0),vel=(1,0))
#zs = simulate_mouvement(N = 50, R_std=noise_std, init_pos=(1,0), init_vel=(1,0.2))
#
#filter = kalman_filter(1, 1, 0, 0, 0, 0)
#print("Initial state: ", filter.x)
#
#for i, z in enumerate(zs):
#    filter.predict()
#    filter.update(z)
#    #print("Step: ", i, "State: ", filter.x)
#filter.plot(y_err=noise_std, x_err=noise_std)
#plt.show()
#
#filter.plot_residuals()
