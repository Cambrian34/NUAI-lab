import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream depth and color frames
config = rs.config()
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Create an OpenCV window to display the video
cv2.namedWindow("RealSense Video", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()

        # Get the color frame
        depth_frame = frames.get_depth_frame()

        # Convert the color frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)

        # Display the image using OpenCV
        cv2.imshow("RealSense Video", depth_colormap)

        # Break the loop if the user presses the 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    # Stop the pipeline
    pipeline.stop()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
