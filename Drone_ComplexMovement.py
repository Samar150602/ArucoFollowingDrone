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
# Kp_x, Ki_x, Kd_x = 0.4, 0.02, 0.2  # Tune these values as needed
# Kp_y, Ki_y, Kd_y = 0.8, 0.05, 0.1 
# Kp_z, Ki_z, Kd_z = 0.4, 0.05, 0.1

# Kp_x, Ki_x, Kd_x = 0.4, 0.02, 0.2  # Tune these values as needed
# Kp_y, Ki_y, Kd_y = 0.4, 0.02, 0.2 
# Kp_z, Ki_z, Kd_z = 0.4, 0.02, 0.2


Kp_x, Ki_x, Kd_x = 0.385, 0.02, 0.25  # Tune these values as needed
Kp_y, Ki_y, Kd_y = 0.385, 0.02, 0.25 
Kp_z, Ki_z, Kd_z = 0.4, 0.02, 0.25

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
            time.sleep(0.0001)
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
                    # print("Distance to Marker ID {}: {:.2f} meters".format(ids[i][0], distance))


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
                    # print("y_control", y_control)
                    print("---------------")
                    print("Error_X", error_x)
                    print("Error_Y", error_y)
                    print("Error_Z", error_z)
                    print("*****************")
                    print("x_control", x_control)
                    print("y_control", y_control)
                    print("z_control", z_control)
                    print("Distance to Marker ID {}: {:.2f} meters".format(ids[i][0], distance))
                    print("---------------")
                    
                    if abs(error_x) < 20:
                        x_control = 0
                    if abs(error_y) < 20:
                        y_control = 0
                    if abs(error_z) < 0.25:
                        z_control = 0
                    
                    
                    # Increase scaling factors
                    scaling_factor_x = 0.0009 #0.0005  # Example increase
                    scaling_factor_y = 0.009
                    # drone.move(-x_control * scaling_factor, z_control * scaling_factor, y_control * scaling_factor, 0)
                    drone.moveAll(-x_control * scaling_factor_x, z_control, y_control * scaling_factor_y)
                    # drone.move(-x_control*0.005, z_control, y_control, 0)    # Do movement
                    
                    
                    # leftRight       = -0.02                                          # Move with 2% speed to the left
                    # backwardForward = 0.0                                           # Move with 10% speed backwards
                    # downUp          = 0.0                                            # Move with 30% speed upward
                    # turnLeftRight   = 0                                              # Turn full speed right
                    # drone.move(leftRight, backwardForward, downUp, turnLeftRight)    # Do movement
            else:
                # print("No markers detected.")
                drone.stop()                   # Drone stops
                time.sleep(0.01)

            
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
