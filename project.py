from computer_vision import *

center =[]
area =[]
object =[]
framerate = 30
time_prev = 0

video_capture = setup_camera(-8)

while True:
    time_elapsed = time.time() - time_prev
    ret, frame = video_capture.read()
    
    
    #obst_centers, obst_contours = computer_vision(frame, 'obstacle')
    #goal_center, goal_contours = computer_vision(frame, 'goal')
    robot_center, robot_contour = computer_vision(frame, 'robot')
    
    
    if (len(robot_contour) > 0):
        print(robot_contour[0][0][0])
        cv2.circle(frame, robot_contour[0][1][0], 6, [0, 0, 255], -1)
    #    print("center",robot_center[0])
    #    print("contour",robot_contour[0][0][0])
        print(get_robot_position(robot_center[0], robot_contour[0]))
    cv2.imshow('Computer Vision', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the capture
video_capture.release() 
cv2.destroyAllWindows()



