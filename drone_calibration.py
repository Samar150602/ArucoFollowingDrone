import time, sys, cv2, numpy as np
import ps_drone

# Startup sequence for the drone
drone = ps_drone.Drone()
drone.startup()

drone.reset()
while drone.getBattery()[0] == -1:
    time.sleep(0.1)

print("Battery: {}%  {}".format(drone.getBattery()[0], drone.getBattery()[1]))

# Configure video
drone.useDemoMode(True)
drone.sdVideo()
drone.frontCam()
CDC = drone.ConfigDataCount
while CDC == drone.ConfigDataCount:
    time.sleep(0.0001)

drone.startVideo()

IMC = drone.VideoImageCount
stop = False
counter = 0

# Chessboard settings
chessboard_size = (8, 6)
square_size = 25  # Adjust based on chessboard units (e.g., mm or cm)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store points
objpoints = []  # 3D points
imgpoints = []  # 2D points

print("Press 'q' to stop early.")

# Collect up to 30 images with a valid chessboard pattern
while counter < 50:
    while drone.VideoImageCount == IMC:
        time.sleep(0.1)
    IMC = drone.VideoImageCount

    frame = drone.VideoImage
    if frame is None:
        continue

    # Convert frame to grayscale
    color_frame = np.copy(frame)
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_frame, chessboard_size, None)

    if ret:
        # Append object points
        objpoints.append(objp)

        # Refine corners and append image points
        corners = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw corners for visualization
        cv2.drawChessboardCorners(color_frame, chessboard_size, corners, ret)
        cv2.imshow("Chessboard Detection", color_frame)

        # Save images for debugging
        cv2.imwrite("frame_{counter:02d}_corners.png", color_frame)
        print("Captured frame {counter + 1} with detected chessboard.")

        counter += 1
    else:
        print("No chessboard detected in this frame.")

    # Display frame and check for 'q' to quit
    cv2.imshow("Current Frame", color_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early termination requested.")
        break

cv2.destroyAllWindows()

# Perform calibration if sufficient points were collected
if len(objpoints) > 0 and len(imgpoints) > 0:
    print("Performing camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

    if ret:
        print("Calibration successful!")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)

        # Save calibration results
        np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeffs=dist)
        print("Calibration data saved to 'camera_calibration.npz'.")
    else:
        print("Calibration failed. Try capturing better images.")
else:
    print("Not enough points for calibration.")
