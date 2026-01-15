import pyrealsense2 as rs
import numpy as np
import cv2
from interbotix_xs_modules.locobot import InterbotixLocobotXS
import math
from ultralytics import YOLO
import time


locobot = InterbotixLocobotXS("locobot_px100", "mobile_px100")
locobot.camera.pan_tilt_move(0, 1.1)
locobot.camera.pan(0)
#locobot.arm.go_to_sleep_pose()

pipeline = rs.pipeline()
config = rs.config()
model = YOLO("/home/locobot/Music/Diwali-Sorter-main/BEST_DUCKS.pt")  # <-- change path if needed
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()

cv2.namedWindow("RealSense Video", cv2.WINDOW_AUTOSIZE)
def pickup(object_name= 'candle', obj_ctr_x=0, obj_ctr_y=0):
	if(object_name == 'candle' or object_name == 'Plant'):
		Z_depth = -0.21
	else:
		Z_depth = -0.189
	
	locobot.arm.go_to_home_pose()
	locobot.arm.set_ee_cartesian_trajectory(x=-0.03, z= Z_depth)
	#locobot.arm.set_ee_cartesian_trajectory(x=0.03)
	#locobot.gripper.close(2.0)
	#time.sleep(0.5)
	#locobot.arm.set_ee_cartesian_trajectory(z=0.2)
	#locobot.arm.go_to_home_pose()
	#locobot.arm.set_ee_cartesian_trajectory(x=-0.12, z= -0.07)
	#time.sleep(0.5)

def hip(object_name= 'candle', obj_ctr_x=0, obj_ctr_y=0):
	t1 = math.atan(-53/256)	
	x_dist = 422-obj_ctr_x
	y_dist = 480-obj_ctr_y
	angle = math.atan(x_dist/y_dist)
	locobot.arm.set_single_joint_position("waist", angle+t1)
	#locobot.gripper.close(2.0)
	#time.sleep(0.5)
	#locobot.arm.set_ee_cartesian_trajectory(z=0.2)
	#locobot.arm.go_to_home_pose()
	#locobot.arm.set_ee_cartesian_trajectory(x=-0.12, z= -0.07)
	#time.sleep(0.5)


def object_identification(object_name):
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	if not depth_frame or not color_frame:
		raise RuntimeError("Could not aquire depth or color data")
		
	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())
	
	results = model(color_image)
	annotated_frame = results[0].plot()
	cv2.imshow("RealSense Video", annotated_frame)
	time.sleep(1)
	
        
	target_class_name = object_name # e.g., 'person', 'car'
	for class_id, class_name in model.names.items():
		print(class_name)
		print(class_id)
		if class_name == target_class_name:
			target_class_id = class_id
			break
	print(target_class_id)
	obj_found = 0
	for r in results: # Iterate through batches if multiple images were processed
		for box in r.boxes:
			class_id = int(box.cls[0])
			#print("Here1")
			if class_id == target_class_id:
				# Extract bounding box coordinates (xyxy format: [x1, y1, x2, y2])
				#print("Here2")
				x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
				confidence = round(box.conf[0].item(), 2)
				obj_center_x = (x1 + x2)/2
				obj_center_y = (y1 + y2)/2
				#print(f"candel center is at {obj_center_x}")
				#print(f"number of pixels away {distance_center}")
				obj_found = 1
				print(f"Detected {target_class_name} at: "
				f"obj_center_x={obj_center_x}, obj_center_y={obj_center_y} "
				f"Confidence={confidence}")
				break
			else:
				obj_found = 0
				obj_center_x = 378
				obj_center_y = 240
	return(obj_found, obj_center_x, obj_center_y)
	# You can storeqq these in a list or perform further actions
	# detected_boxes.appeqnd({'coords': [x1, y1, x2, y2], 'confidence': confidence})
	# You can store these in a list or perform further actions
	# detected_boxes.append({'coords': [x1, y1, x2, y2], 'confidence': confidence})
	############################################################

def drop():
	locobot.arm.go_to_home_pose()
	locobot.arm.set_ee_cartesian_trajectory(z=-0.2)
	locobot.gripper.open(2.0)
	locobot.arm.set_ee_cartesian_trajectory(z=0.2)
	locobot.arm.go_to_sleep_pose()


pickup('diya', 378, 560)
while True:
	found, obj_x, obj_y = object_identification('candle')
	print(obj_x, obj_y)
	#hip('duck', obj_x, obj_y)
	#drop()
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
	
