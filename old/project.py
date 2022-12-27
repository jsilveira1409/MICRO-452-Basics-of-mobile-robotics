import numpy as np
import math
import time
from computer_vision import *
from dijkstra import compute_shortest_path
from kalman import kalman_filter
from test import *

#read_camera(-7)
fig, ax = plt.subplots(figsize=(12,12))
cv_successful, obst, robot, goal, frame = cv_start(show_image= True, exposure=-7)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
obst = format_contour(obst)

metric_robot = pixel_to_metric(robot[:2])
print("cv successful: ", cv_successful)
print("robot : ", robot , metric_robot)
print("goal  : ", goal)
print("obst  : ", obst)


if cv_successful:
    # execute dijkstra
    start = np.array([robot[0],robot[1]])
    print("start",start)
    print("goal", goal)
    path = compute_shortest_path(obst, start, goal)
    path = np.rint(path).astype(int)

    position_estimate =  start
    theta_estimate = 0
    print(path)
    path = np.delete(path, 0, 0)
    for tar in path:
        pos_hist, thet_hist = pathing(position_estimate, theta_estimate, tar, min_distance=20)
        if pos_hist is not None and thet_hist is not None:
            position_estimate = pos_hist[-1]
            theta_estimate = thet_hist[-1]
        
        plt.plot([p[0] for p in pos_hist], [p[1] for p in pos_hist])
        plt.quiver(pos_hist[-1][0], pos_hist[-1][1], np.cos(theta_estimate), np.sin(theta_estimate), color='red')
        plt.scatter(tar[0], tar[1], color = 'red', marker = 'x')
        #plt.set_xlim(-500, 2000)
        #plt.set_ylim(-1500, 1500)
        #set_speed(0,0)
        #time.sleep(1)
        
    print(position_estimate, theta_estimate)
    draw_path(frame, path) 


#plt.savefig("path.png")
#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#
#if cv_successful:
#    float_formatter = "{:.2f}".format
#    np.set_printoptions(formatter={'float_kind':float_formatter})
#    filter = kalman_filter(robot[0], 0, robot[0], 0, robot[2], 0)
#    print("Initial state: ", filter.x)
plt.show()