import time
import ps_drone
import cv2  
import numpy as np


# Define camera matrix and distortion coefficients (from camera calibration)
# camera_matrix = np.array([[fx,  0, cx],
#                           [ 0, fy, cy],
#                           [ 0,  0,  1]])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Update with actual distortion coefficients
camera_matrix = np.array([[574.15552006,   0.        , 341.30393474],
                          [0.        , 569.59685087, 168.60591048],
                          [0.        ,   0.        ,   1.        ]], dtype=np.float32)

# Distortion coefficients (k1, k2, p1, p2, k3)
dist_coeffs = np.array([-5.55051140e-01,  4.96057547e-01, -4.68844974e-04,
        -1.25603122e-03, -1.00136772e+00], dtype=np.float32)


# PID Parameters for X and Y and Z
Kp_x, Ki_x, Kd_x = 1, 0.05, 0.1  # Tune these values as needed
Kp_y, Ki_y, Kd_y = 0.8, 0.05, 0.1 
Kp_z, Ki_z, Kd_z = 1, 0.05, 0.2


# Initialize PID variables
prev_error_x, integral_x = 0.0, 0.0
prev_error_y, integral_y = 0.0, 0.0
prev_error_z, integral_z = 0.0, 0.0

# Initialize the drone
drone = ps_drone.Drone()
drone.startup()

# Reset the drone and wait for it to stabilize
drone.reset()
while (drone.getBattery()[0] == -1):      
    time.sleep(0.1)
print("Battery: "+str(drone.getBattery()[0])+"%  "+str(drone.getBattery()[1]))
drone.useDemoMode(True)

# Video setup
drone.setConfigAllID()
drone.sdVideo()
drone.frontCam()
CDC = drone.ConfigDataCount
while CDC == drone.ConfigDataCount:
    time.sleep(0.0001)

drone.startVideo()

IMC = drone.VideoImageCount

# Load ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters_create()

# Function to display live video feed
def camera_feed():
    global IMC, prev_error_x, integral_x, prev_error_y, integral_y, prev_error_z, integral_z
    drone.setSpeed(0.01)  # Set a safe speed
    last_time = time.time()  # Initialize time for PID calculations

    while True:
        # Wait until a new video frame is available
        while drone.VideoImageCount == IMC: 
            time.sleep(0.01)
        IMC = drone.VideoImageCount

        # Get the current video frame
        frame = drone.VideoImage
        if frame is not None:
            # Convert the frame for OpenCV processing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

            # Draw detected markers and process them
            if ids is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # print("Detected IDs:", ids)
                
                # Define the actual size of the marker (in meters or any consistent unit)
                marker_length = 0.1  # Example: 10 cm

                # Estimate the pose of the markers
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

                # Extract frame dimensions for camera center
                height, width = frame.shape[:2]
                camera_centerX = width / 2
                camera_centerY = height / 2
                # print("Camera Center: ({:.2f}, {:.2f})".format(camera_centerX, camera_centerY))

                # Iterate through each detected marker
                for i, corner in enumerate(corners):
                    # Extract the array of points (the corners of the marker)
                    points = corner[0]
                    if ids[i][0] != 50:
                        continue

                    # Calculate the center of the detected marker
                    marker_centerX, marker_centerY = np.mean(points, axis=0)
                    # print("Marker ID {} AR Center: ({:.2f}, {:.2f})".format(ids[i][0], marker_centerX, marker_centerY))

                    # Draw a circle at the center of the marker
                    cv2.circle(frame, (int(marker_centerX), int(marker_centerY)), radius=5, color=(0, 255, 0), thickness=-1)


                    # Calculate distance from the camera
                    tvec = tvecs[i][0]  # Translation vector for this marker
                    distance = np.linalg.norm(tvec)  # Euclidean distance
                    print("Distance to Marker ID {}: {:.2f} meters".format(ids[i][0], distance))


                    # Calculate the error between camera center and marker center
                    error_x = camera_centerX - marker_centerX
                    error_y = camera_centerY - marker_centerY
                    error_z = distance - 1.75
                    
                    # print("---------------")
                    # print("Error_X", error_x)
                    # print("Error_Y", error_y)
                    # print("---------------")
                    
                    
                    # Time difference
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time
                    
                    # PID control for X (left-right)
                    integral_x += error_x * dt
                    derivative_x = (error_x - prev_error_x) / dt if dt > 0 else 0
                    x_control = Kp_x * error_x + Ki_x * integral_x + Kd_x * derivative_x
                    prev_error_x = error_x


                    # PID control for Y (up-down)
                    integral_y += error_y * dt
                    derivative_y = (error_y - prev_error_y) / dt if dt > 0 else 0
                    y_control = Kp_y * error_y + Ki_y * integral_y + Kd_y * derivative_y
                    prev_error_y = error_y
                    
                    
                     # PID control for Z (Forward-Backward)
                    integral_z += error_z * dt
                    derivative_z = (error_z - prev_error_z) / dt if dt > 0 else 0
                    z_control = Kp_z * error_z + Ki_z * integral_z + Kd_z * derivative_z
                    prev_error_z = error_z


                    
                    
                    # print("---------------")
                    # print("x_control", x_control)
                    print("y_control", y_control)
                    print("---------------")
                    print("Error_X", error_x)
                    print("Error_Y", error_y)
                    print("Error_Z", error_z)
                    print("*****************")
                    # print("x_control", x_control)
                    # print("z_control", z_control)
                    print("---------------")
                    
                    # Control the drone using computed signals
                    if abs(error_x) > 10:  
                        if x_control > 0:
                            drone.moveLeft(x_control*0.0009)  
                            print("left")
                        else: 
                            drone.moveRight(-x_control*0.0009)
                            print("right")
                    
                    else:
                        drone.stop()                   # Drone stops
                        time.sleep(0.1)
                    
                    # if abs(error_y) > 10:  
                    #     if y_control > 0:
                    #         drone.moveUp(y_control*0.009)  
                    #         print("top")
                    #     else: 
                    #         drone.moveDown(-y_control*0.009)
                    #         print("bottom")
                            
                    # else:
                    #     drone.stop()                   # Drone stops
                    #     time.sleep(0.1)
                    
                    
                    # if abs(error_z) > 0.25:  
                    #     if z_control > 0:
                    #         print("Forward")
                    #         drone.moveBackward(-z_control) 
                    #     else: 
                    #         print("Backward")
                    #         drone.moveForward(z_control) 
                    
                    # else:
                    #     drone.stop()                   # Drone stops
                    #     time.sleep(0.1)
                            
                    # if abs(error_y) > 10:
                    #     drone.moveUp(y_control) if y_control > 0 else drone.moveDown(-y_control)


                    # # Determine the relative position of the marker
                    # if error_x > 0:
                    #     position_x = "Right"  
                    # else: 
                    #     position_x = "Left"
                        
                    # if error_y > 0:
                    #     position_y = "Top" 
                    # else: 
                    #     position_y = "Bottom"

                        
                    
                    # if distance > 5:
                    #     print("Too far, come closer")
                    #     error_z = "far"
                    # else:
                    #     print("Too close, go backward")
                    #     error_z = "close"
                        
                    # # print('Aruco Marker is on the "{}" of the Camera'.format(position_x))
                    # # print('Aruco Marker is on the "{}" of the Camera'.format(position_y))
                    # print('Aruco Marker is "{}" from the Camera'.format(error_z))
                        
                    
                    # Annotate the distance on the frame
                    # cv2.putText(frame, "ID: {} Dist: {:.2f}m".format(ids[i][0], distance),
                    #             (int(marker_centerX), int(marker_centerY) - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
            else:
                # print("No markers detected.")
                drone.stop()                   # Drone stops
                time.sleep(0.1)

            
            # Display the result
            cv2.imshow('Detected ArUco Markers', frame)
    
            # print("inside while loop!!!!")

            # Wait for key press to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# # Drone control
drone.setSpeed(0.1)  # Set default moving speed to 0.1

# Takeoff the drone
drone.takeoff()      
time.sleep(3)        # Wait for 3 seconds of stabilization
print("took off")
# Start the camera feed in a separate function to run continuously
camera_feed()
print("camera output")
drone.stop()                   # Drone stops
time.sleep(2)

# Automatically land the drone after 3 seconds of stabilization and feed running
drone.land()          # Land the drone after stabilization period
time.sleep(2)         # Wait a bit to ensure landing completes

# Clean up resources
cv2.destroyAllWindows()
