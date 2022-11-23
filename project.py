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
    
    
    obst_centers, obst_contours = computer_vision(frame, 'obstacle')
    #goal_center, goal_contours = computer_vision(frame, 'goal')
    #robot_center, robot_contour = computer_vision(frame, 'robot')
    
    
    cv2.imshow('Computer Vision', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the capture
video_capture.release() 
cv2.destroyAllWindows()



