import numpy as np
import math
import time
from computer_vision import *
from dijkstra import compute_shortest_path
from kalman import kalman_filter

#read_camera(-7)
cv_successful, obst, robot, goal, frame = cv_start(show_image= True)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
obst = format_contour(obst)

metric_robot = pixel_to_metric(robot[:2])
print("cv successful: ", cv_successful)
print("robot : ", robot , metric_robot)
print("goal  : ", goal)


if cv_successful:
    # execute dijkstra
    start = [robot[0],robot[1]]
    print(start)
    print(goal)
    path = compute_shortest_path(obst, start, goal)
    path = np.rint(path).astype(int)

    print("path")
    for p in path:
        print(p)

    draw_path(frame, path)

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if cv_successful:
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    filter = kalman_filter(robot[0], 0, robot[0], 0, robot[2], 0)
    print("Initial state: ", filter.x)
plt.show()