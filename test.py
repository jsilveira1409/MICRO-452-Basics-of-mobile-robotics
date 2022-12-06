import numpy as np
import time


def get_angle_between(vec1, vec2):
    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.dot(vec1_unit, vec2_unit))


def pathing(position_estimate, theta_estimate, target, min_distance = 50):
    position_history = []
    theta_history = []
    direction = np.array([np.cos(theta_estimate), np.sin(theta_estimate)])
    # outputs, in mm/s
    speed_l = 0
    speed_r = 0
    # in radians
    angle_tolerance = 0.17
    # time step(s)
    dt = 0.01
    # counts nb of steps
    i = 0
    
    distance = np.sqrt((target[0]-position_estimate[0])**2 + (target[1]-position_estimate[1])**2)
    while distance > min_distance:

        depl_left = speed_l * dt * 3.2
        depl_right = speed_r * dt * 3.2
        depl_center = (depl_right + depl_left)/2
        
        # divided by the wheel distance(in mm) to get the angle(rad), trigonometric sense
        theta = (depl_left - depl_right) / 100
        if theta != 0:
            theta_estimate = (theta_estimate + theta) % (2*np.pi)
            
        depl_x = depl_center * np.cos(theta_estimate) 
        depl_y = depl_center * np.sin(theta_estimate)
        position_estimate = position_estimate + np.array([depl_x, depl_y])
        
        # relative target vector with respect to the robot position
        target_direction = target - position_estimate
        direction = np.array([np.cos(theta_estimate), np.sin(theta_estimate)])
        alpha = get_angle_between(target_direction, direction)
        if (alpha <= angle_tolerance and alpha >= 0) or (alpha >= -angle_tolerance and alpha < 0) :
            # move forward, mm/s
            speed_l = 50 - 40 * np.sin(alpha)
            speed_r = 50 + 40 * np.sin(alpha)
        elif alpha > angle_tolerance:            
            speed_l = -50
            speed_r = 50 
        elif alpha < - angle_tolerance:
            speed_l = 50
            speed_r = -50
    
        distance = np.sqrt((target[0]-position_estimate[0])**2 + (target[1]-position_estimate[1])**2)

        if i > 1000000 :
            print(i, distance)
            break
        #print(position_estimate, distance) 
        position_history.append(position_estimate)
        theta_history.append(theta_estimate)
        #set_speed(speed_l, speed_r)
        #time.sleep(dt)
        
        i += 1

    return position_history, theta_history


def pathing_bis(position_estimate, theta_estimate,  target, speed_l, speed_r, dt = 0.01):
    angle_tolerance = 0.2

    depl_left = speed_l * dt * 3.2
    depl_right = speed_r * dt * 3.2
    depl_center = (depl_right + depl_left)/2
    theta = (depl_left - depl_right) / 100
    
    if theta != 0:
        theta_estimate = (theta_estimate + theta) % (2*np.pi)
    
    depl_x = depl_center * np.cos(theta_estimate) 
    depl_y = depl_center * np.sin(theta_estimate)
    position_estimate = position_estimate + np.array([depl_x, depl_y])
    
    target_direction = target - position_estimate
    direction = np.array([np.cos(theta_estimate), np.sin(theta_estimate)])
    alpha = get_angle_between(target_direction, direction)
    #print(alpha)

    if (alpha <= angle_tolerance and alpha >= 0) or (alpha >= -angle_tolerance and alpha < 0) :
        # move forward, mm/s
        speed_l = 50 - 40 * np.sin(alpha)
        speed_r = 50 + 40 * np.sin(alpha)
    elif alpha > angle_tolerance:            
        speed_l = -50
        speed_r = 50 
    elif alpha < - angle_tolerance:
        speed_l = 50
        speed_r = -50
    distance = int(np.sqrt((target[0]-position_estimate[0])**2 + (target[1]-position_estimate[1])**2))
    #set_speed(speed_l, speed_r)
    #time.sleep(dt)

    return distance, position_estimate, theta_estimate, speed_l, speed_r
    


        
        
