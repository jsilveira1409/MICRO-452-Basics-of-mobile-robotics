from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse
from scipy.linalg import block_diag


R_std = 1
R_theta_x = R_theta_y = 0.1
R_theta_theta = 0.1


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


def SecondOrderKF(R_std, Q, dt, P=100):

    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.x = np.zeros(3)
    kf.P[0, 0] = P
    kf.P[1, 1] = 1
    kf.P[2, 2] = 1
    kf.R *= R_std**2
    kf.Q = Q_discrete_white_noise(3, dt, Q)
    kf.F = np.array([[1., dt, .5*dt*dt],
                     [0., 1.,       dt],
                     [0., 0.,       1.]])
    kf.H = np.array([[1., 0., 0.]])
    return kf


def kalman_setup(dt = 1.0, x0 = 1.0, y0=1.0, th0=0.):
    # 3 state variables to track : x, y, theta, vx, vy, omega
    # 3 measurements : x, y, theta
    filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=2)
    # STEP 1
    # state transition matrix
    filter.F = np.array(    [[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    # STEP 2
    # no initial control input
    filter.u = np.array([0, 0, 0])
    # control input function
    filter.B = np.eye(3)

    # STEP 3
    # measurement function. z = (zx, zy, ztheta)
    filter.H = np.eye(3)

    # STEP 4
    # we assume that the error is independant in x and y, therefore no correlation 
    # between the two. However, we assume that the error is correlated in theta as 
    # it depends on the x and y translation.
    filter.R = np.array([[    R_std,         0,     R_theta_x],
                         [        0,     R_std,     R_theta_y],
                         [R_theta_x, R_theta_y, R_theta_theta]])

    # STEP 5
    # process noise covariance matrix

    # values for the Q matrix
    # q_xth, q_xvx, q_xvth, q_vxth, qvxvth
    # q_yth, q_yvy, q_yvth, q_vyth, qvyvth
    # q_thvth
    filter.Q = np.array([[1,0,0],
                         [0,0,0],
                         [0,0,0]])

    # STEP 6
    # initial state, the robot is not moving at the beginning.
    # x, y, and theta is given by CV, vx, vy and vtheta are 0.
    filter.x = np.array([[0, 0, 0]]).T
    # initial covariance matrix 
    # TODO: not sure how to set this up, as it is a random guess, i think
    filter.P = [np.eye(3) * 100.]
    return filter


def simulate_mouvement(N = 10, init_pos = (0,0), init_vel = (1,1)):
    sensor = sensor_position(init_pos, init_vel, noise_std=R_std)
    zs = np.array([sensor.read() for _ in range(N)])
    return zs

def plot_confidence(mu, cov, std = 3):
    for x, P in zip(mu, cov):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]], 
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        plot_covariance_ellipse(mean, cov=cov, fc='g', std=std, alpha=0.2)


filter = SecondOrderKF(0.1, 0.1, 1.0)
#zs = simulate_mouvement()
zs = []

(mu, cov, _, _) = filter.batch_filter(zs = zs)
#plot_confidence(mu,cov)
##
###plot results
#plt.plot(mu[:, 0], mu[:, 2], label='filter', marker='o', color='r')
##
#plt.plot(zs[:, 0], zs[:, 1], label='measurements', color='g')
#plt.legend(loc=2)
##
#plt.grid()
#plt.show()



