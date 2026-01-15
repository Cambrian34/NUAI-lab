import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


model = YOLO("best_v11s.pt")  # <-- change path if needed


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)


try:
	for i in range(60):
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		
		if not depth_frame or not color_frame:
			raise RuntimeError("Could not aquire depth or color data")
			
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		
	cv2.imwrite("color_im.png", color_image)
	cv2.imwrite("depth_color.png", depth_colormap)
	
	print("Images captured and saved")

finally:
	pipeline.stop()
