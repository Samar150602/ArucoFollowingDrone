import time
import ps_drone
import cv2  # OpenCV for processing video frames

# Initialize the drone
drone = ps_drone.Drone()
drone.startup()

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

print("Press any key to exit the video feed...")

IMC = drone.VideoImageCount
# stop = False

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

        # Add a bounding box (example coordinates and size)
        x, y, w, h = 100, 100, 200, 150  # Define top-left corner and size of the box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, 2px thick

        # Optionally, add text
        cv2.putText(frame, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the modified frame
        cv2.imshow("Drone Live Feed with Bounding Box", frame)
        
        cv2.waitKey(1)

        # # Check for key press to stop
        # key = cv2.waitKey(1) & 0xFF
        # if key != 255:  # Press any key to exit
        #     stop = True

# Clean up
# cv2.destroyAllWindows()
