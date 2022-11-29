from computer_vision import *
from dijkstra import compute_shortest_path
center =[]
area =[]
object =[]
framerate = 30
time_prev = 0


#start by seting up the computer vision
obst, robot, goal, frame= cv_start(exposure=-8, show_image= True)

obst = format_contour(obst)
print("obstacles")
for o in obst:
    print(o)

print("robot")
print(robot)
print("goal")
print(goal)

# execute dijkstra
start = [robot[0],robot[1]]
path = compute_shortest_path(obst, start, goal)
path = np.rint(path).astype(int)

print("path")
for p in path:
    print(p)

draw_path(frame, path)

cv2.imshow('frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
