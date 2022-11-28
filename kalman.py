from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
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
        self.R = np.array([ [10,  0,    0],
                            [0,  10,    0],
                            [0,   0,   10]])
        # process noise covariance 
        self.Q = 0.01 * np.array([[0.25, 0.5,    0,    0,    0,   0],
                                 [0.5,    1,    0,    0,    0,   0],
                                 [  0,    0, 0.25,  0.5,    0,   0],
                                 [  0,    0,  0.5,    1,    0,   0],
                                 [  0,    0,    0,    0, 0.25, 0.5],
                                 [  0,    0,    0,    0,  0.5,   1]])
        # state covariance matrix
        # since we are uncertain about the initial state,
        # we set the initial state covariance matrix to a high value
        # which is equivalent of increasing the uncertainty
        self.P = 500 * np.array([  [1, 0, 0, 0, 0, 0],
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
        self.P_post = copy.deepcopy(self.P)
        self.x_copy = copy.deepcopy(self.x)
        self.z = copy.deepcopy(z)

        return self.x, self.P
    

    # processes a sequence of measurements
    def filter(self, z):
        nb_measurements = z.shape[1]
        dim_x = self.x.shape[0]
        means = np.zeros((nb_measurements, dim_x))
        means_prediction = np.zeros((nb_measurements, dim_x))        

        cov = np.zeros((nb_measurements, dim_x, dim_x))
        cov_prediction = np.zeros((nb_measurements, dim_x, dim_x))

        for i, (z, F, Q, H, R, B, u) in enumerate(zip(self.z, self.F, self.Q, self.H, self.R, self.B, self.u)):
            self.update(z)
            means[i, :] = self.x
            cov[i, :, :] = self.P

            self.predict()
            means_prediction[i, :] = self.x
            cov_prediction[i, :, :] = self.P

        return means, cov, means_prediction, cov_prediction



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



float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

sensor = sensor_position()
zs = simulate_mouvement()

filter = kalman_filter(1, 1, 1, 1, 1, 1)
means, cov, pred_means, pred_cov = filter.filter(zs)


