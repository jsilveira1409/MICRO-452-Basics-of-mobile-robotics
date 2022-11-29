import numpy as np
from tdmclient import ClientAsync

def avoid_obstacle(x,y):
    dist_l = 2*prox.horizontal[0] + prox.horizontal[1]
    dist_r = 2*prox.horizontal[4] + prox.horizontal[3]
    motor.left.target = motor.left.target + dist_l // 10
    motor.right.target = motor.right.target + dist_r // 10