# states = [x, y, theta]  u = [l-speed, r_speed]
import numpy as np
import time



#constant parameters
PX_TO_MM = 1
TS_2_MMPERS = 140/500 
THYMIO_WIDTH = 95 #in mm
SPEED_VARIANCE = 0.0001 #in mm^2/s^2
CAMERA_POSITION_VARIANCE = 0.0001 #in mm^2
THYMIO_ORI_length = 60 #in mm
CAMERA_ANGLE_VARIANCE = 0.0001#1/(THYMIO_ORI_length*4)**2 #in rad^2


#----------------------------- Kalman initialization -----------------------------
def kalman_init(left_speed, right_speed):
    x_initial = 0.3
    y_initial = 0.3
    theta_initial = 0.5
    X = np.array([[x_initial], [y_initial], [theta_initial]], dtype='float')
    Number_of_states = len(X)
    U = np.array([[left_speed], [right_speed]], dtype='float')
    #We syppose we are uncertain about the initial states of the robot
    P_Sigma = np.diag([1000, 1000, 1000])
    return X, U, P_Sigma

#----------------------------- States prediction -----------------------------

def predict(X, U, P_Sigma, prev_time,robot_in_sight):
    
    start_time = time.time()
    
    if (prev_time == 0):
        dt = 0.05
    else : 
        dt = start_time - prev_time
    

    Number_of_states = len(X)   #3
    Control_length = len(U)     #2

    #Defining the system dynamics : A= 3x3, B= 3x2
    
    A = np.eye(Number_of_states)
    
    B = np.array([[0.5 * dt * np.cos(X[2]),     0.5 * dt * np.cos(X[2])], 
                  [0.5 * dt * np.sin(X[2]),     0.5 * dt * np.sin(X[2])],
                  [ dt / THYMIO_WIDTH,          -dt / THYMIO_WIDTH]], dtype='float')
 
    #States prediction 
    
    X = A.dot(X) + B.dot(U) 
    
    #Defining the Q matrix : covariance matrix of the predicted states : depend on the variance of the speed. shape = 2x2 
    
    Q = np.eye(Control_length) *  SPEED_VARIANCE
 
    #Sigma prediction : predicted states covariance. shape : 3x3
    
    P_Sigma = P_Sigma + B.dot(Q).dot(B.T)
    
    return X, P_Sigma, start_time

#----------------------------- Measurements update -----------------------------

def update(X, Z, P_Sigma, robot_in_sight):
    
    #Defining the R matrix : covarience of the measurements : from the camera. shape = 3x3
    R = np.diag([CAMERA_POSITION_VARIANCE, CAMERA_POSITION_VARIANCE, CAMERA_ANGLE_VARIANCE]) 
    
    #Defining H matrix : measurements matrix. shape = 3x3
    Number_of_states = len(X) 
    if robot_in_sight :
        H = np.eye(Number_of_states)
    else : 
        H = np.zeros((Number_of_states, Number_of_states))
    
    #Innovation calculation : difference between measurements and prediction 
    
    I = Z - X
    
    #S matrix calculation
    
    S = H.dot(P_Sigma).dot(H.T) + R
    
    #Kalman gain calculation
    
    K = P_Sigma.dot(H.T).dot(np.linalg.inv(S))
    
    #States update
    
    X = X + K.dot(I)
    
    #States covariance update 
    
    P_Sigma = P_Sigma - K.dot(H).dot(P_Sigma)
    
    return X, P_Sigma

#----------------------------- Kalman filtering -----------------------------

def Kalman(X, U, Z, P_Sigma, prev_time, robot_in_sight):

    X, P_Sigma, start_time = predict(X, U, P_Sigma, prev_time, robot_in_sight) 
    X_predicted = np.array([[X[0]],[X[1]]])
    
    X, P_Sigma = update(X, Z, P_Sigma, robot_in_sight)
    
    return X, P_Sigma, start_time, X_predicted
    
    
        
    
    
    
    
    