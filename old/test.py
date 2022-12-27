import numpy as np
import math
import time
from computer_vision import *
from dijkstra import compute_shortest_path
from kalman import kalman_filter

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def get_angle_between(vec1, vec2):
    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.dot(vec1_unit, vec2_unit))

def pathing_bis(node, position_estimate, theta_estimate,  target, speed_l, speed_r, dt = 0.01):
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
    
    node.send_set_variables(motors(int(speed_l),int(speed_r)))

    return distance, position_estimate, theta_estimate, speed_l, speed_r


async def motion_control(node,path):
    min_distance = 10
    
    #filter = kalman_filter(state_estimate[0], state_estimate[1], state_estimate[2], state_estimate[3], state_estimate[4], state_estimate[5])
    position_estimate = [0,0]
    theta_estimate = 0
    speed_l = speed_r = 0
    theta_estimate = 0
    position_history = []
    theta_history = []
    fig, ax = plt.subplots(figsize=(8,8))
    dt = 0.1


    for target in path:
        distance = np.sqrt((target[0] - position_estimate[0])**2 + (target[1] - position_estimate[1])**2)
        position_history = []
        theta_history = []
        while (distance > min_distance):
            distance, position_estimate, theta_estimate, speed_l, speed_r = pathing_bis(node, position_estimate, theta_estimate, target, speed_l, speed_r, dt = dt)   
            position_history.append(position_estimate)
            theta_history.append(theta_estimate)
            node.send_set_variables(motors(int(speed_l),int(speed_r)))    
            
        ax.plot([p[0] for p in position_history], [p[1] for p in position_history])
        ax.quiver(position_history[-1][0], position_history[-1][1], np.cos(theta_estimate), np.sin(theta_estimate), color='red')
        ax.scatter(target[0], target[1], color = 'red', marker = 'x')


    #ax.set_xlim(-5, 1000)
    #ax.set_ylim(-1500, 1500)
    ax.grid()
    fig.show()
    node.send_set_variables(motors(0,0))    

