import numpy as np
import matplotlib.pyplot as plt
from tdmclient import ClientAsync

motor_left_target = 100
motor_right_target = 100

# distance = -339.47*(sensor_value) + 5739.9

@onevent
def prox():
    global prox_horizontal, motor_left_target, motor_right_target
    print(1)
    # acquisition from the proximity sensors to detect obstacles
    motor_left_target = 100 + 2 * (prox_horizontal[0] // 100)
    motor_right_target = 100 + 2 * (prox_horizontal[4] // 100)
    leds_top = [30,30,30]