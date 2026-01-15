#Finds object(s) by moving camerra and uses the depth senser to approximate the distance to it.

import torch
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
import numpy as np
from interbotix_xs_modules.locobot import InterbotixLocobotXS


class ObjFinder:
    def __init__(self):
        # Initialize robot (InterbotixLocobotXS calls rospy.init_node)
        self.robot = InterbotixLocobotXS(robot_model="locobot_px100")
        #self.arm = self.robot.arm()
        #self.gripper = self.robot.gripper()
        self.cmd_pub = rospy.Publisher("/mobile_base/cmd_vel", Twist, queue_size=10)

        self.current_tilt_cmd = 0   # starting tilt, same as scan_camera2
        # CvBridge
        self.bridge = CvBridge()
        self.cv_image = None
        self.Obj_found = False
        self.rate = rospy.Rate(10)

        # Load YOLOv5 model
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #rospy.loginfo("YOLO loaded.")

        # Subscribe to the RealSense topic
        cam_topic = "/camera/color/image_raw"
        rospy.Subscriber(cam_topic, Image, self.image_callback)
        rospy.loginfo(f"Subscribed to camera topic: {cam_topic}")

        # Subscribe to common depth topics (accept whichever is published).
        # Depth callback converts to meters (float32) and stores in `self.depth_image`.
        depth_topics = [
            "/camera/depth/image_rect_raw",
            "/camera/aligned_depth_to_color/image_raw",
            "/camera/depth/image_raw",
        ]
        self.depth_image = None
        for dt in depth_topics:
            try:
                rospy.Subscriber(dt, Image, self.depth_callback)
                rospy.loginfo(f"Listening for depth on: {dt}")
            except Exception:
                pass

    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")

    def depth_callback(self, msg):
        # Convert depth image into meters (float32). Handle common encodings.
        try:
            # Try 32FC1 first
            depth = None
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            except Exception:
                # Some cameras publish uint16 in millimeters
                depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth = depth_u16.astype(np.float32) / 1000.0

            # Replace zeros with NaN for proper averaging later
            depth = depth.astype(np.float32)
            depth[depth == 0] = np.nan
            self.depth_image = depth
        except Exception as e:
            rospy.logwarn(f"Depth conversion failed: {e}")
    """
    def visualize_detections(self, results):
        
        if self.cv_image is None:
            return

        img = self.cv_image.copy()
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Debug", img)
        cv2.waitKey(1)
    """
    """
    def scan_camera(self):
        
        tilt_angles = np.linspace(0, 1, 5)
        pan_angles = np.linspace(-1.04, 1.04, 7)

        rospy.loginfo("Scanning for object...")

        while not self.Obj_found and not rospy.is_shutdown():
            if self.cv_image is None:
                rospy.logwarn("Waiting for image...")
                rospy.sleep(0.1)
                continue

            for tilt in tilt_angles:
                try:
                    self.robot.camera.tilt(tilt)
                    self.current_tilt_cmd = tilt
                except Exception as e:
                    rospy.logwarn(f"Camera tilt failed in scan_camera2: {e}")
                for pan in pan_angles:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    self.robot.camera.pan(pan)
                    rospy.sleep(0.4)

                    results = self.model(self.cv_image)
                    self.visualize_detections(results)
                    

                    for *box, conf, cls in results.xyxy[0]:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        cls_name = self.model.names[int(cls)]

                        if cls_name == "cell phone":
                            rospy.loginfo("Object detected!")
                            self.Obj_found = True
                            return

        rospy.loginfo("Finished scanning.")
    """
    def scan_camera2(self):
        """
        Sweeps the camera to find a red box, and centers the camera on the object's centroid
        by adjusting pan/tilt based on the centroid's pixel location in the image.
        """
        tilt_angles = np.linspace(0, 1, 5)
        pan_angles = np.linspace(-1.04, 1.04, 7)

        rospy.loginfo("Scanning for object...")

        while not self.Obj_found and not rospy.is_shutdown():
            if self.cv_image is None:
                rospy.logwarn("Waiting for image...")
                rospy.sleep(0.1)
                continue

            for tilt in tilt_angles:
                self.robot.camera.tilt(tilt)
                for pan in pan_angles:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    self.robot.camera.pan(pan)
                    rospy.sleep(0.4)

                    img = self.cv_image.copy()
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 120, 70])
                    upper_red1 = np.array([10, 255, 255])
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    lower_red2 = np.array([170, 120, 70])
                    upper_red2 = np.array([180, 255, 255])
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    mask = mask1 + mask2

                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 1000:
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

                                # Calculate error from image center
                                error_x = cx - img.shape[1] // 2
                                error_y = cy - img.shape[0] // 2

                                # Map pixel error to pan/tilt adjustment (tune gain as needed)
                                pan_adjust = np.clip(pan - 0.002 * error_x, -1.04, 1.04)
                                tilt_adjust = np.clip(tilt - 0.002 * error_y, 0, 1)
                                try:
                                    self.robot.camera.pan(pan_adjust)
                                except Exception as e:
                                    rospy.logwarn(f"Camera pan failed in scan_camera2: {e}")
                                try:
                                    self.robot.camera.tilt(tilt_adjust)
                                    self.current_tilt_cmd = float(tilt_adjust)
                                except Exception as e:
                                    rospy.logwarn(f"Camera tilt failed in scan_camera2: {e}")
                                rospy.loginfo(f"Centering camera: pan={pan_adjust:.2f}, tilt={tilt_adjust:.2f}")

                            cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
                            cv2.imshow("Contours", img)
                            cv2.waitKey(1)
                            rospy.loginfo("Object detected and centered!")
                            self.Obj_found = True
                            return
                    cv2.imshow("Red Detection", mask)
                    cv2.waitKey(1)
    
    def approach_obj(self):
        """
        Atempts to approach the object
        
        :param self: Description
        """
        rospy.loginfo("Approaching object...")
        # Use color-based detection (red mask + contours) to approach the
        # object. This mirrors `scan_camera2`'s detection strategy and
        # centers the robot on the centroid of the largest red contour.
        target_area = 20000  # area at which we consider the robot 'close'
        kp_ang = 0.003      # angular gain (tuned experimentally)
        lin_speed = 0.12    # forward speed while approaching
        distance_threshold = 0.40  # meters: stop when closer than this (if depth available)
        consec_needed = 3  # require this many consecutive frames of large area (fallback)
        consec_count = 0

        while not rospy.is_shutdown():
            if self.cv_image is None:
                rospy.sleep(0.05)
                continue

            img = self.cv_image.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                rospy.loginfo("No red contours found — stopping.")
                self.cmd_pub.publish(Twist())
                break

            # choose largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < 500:  # lost or nearly lost
                rospy.loginfo("Target likely lost — attempting reacquisition.")
            
                # stop any motion
                self.cmd_pub.publish(Twist())
            
                # Update last seen centroid + area if possible
                M_tmp = cv2.moments(largest)
                if M_tmp.get("m00", 0) != 0:
                    cx_tmp = int(M_tmp["m10"] / M_tmp["m00"])
                    cy_tmp = int(M_tmp["m01"] / M_tmp["m00"])
                    self.last_seen_cx = cx_tmp
                    self.last_seen_cy = cy_tmp
                    self.last_seen_area = area
            
                try:
                    current_tilt = self.current_tilt_cmd
                    rospy.loginfo(f"[DEBUG] Current camera tilt: {current_tilt:.3f}")

                except:
                    current_tilt = None
                    rospy.logwarn("[DEBUG] Could not get tilt")
                
                TILT_STEP = 0.25           # much larger tilt per cycle
                TILT_MAX  = 1.0            # maximum tilt value (camera downward limit)
                
                # Unconditional sweep: always try tilting more downward first
                if current_tilt is not None and current_tilt < TILT_MAX:
                    new_tilt = min(TILT_MAX, current_tilt + TILT_STEP)
                    rospy.loginfo(f"[SWEEP] Hard tilt down → {new_tilt:.2f}")
                    try:
                        self.robot.camera.tilt(new_tilt)
                        self.current_tilt_cmd = float(new_tilt)
                    except Exception as e:
                        rospy.logwarn(f"Camera tilt failed: {e}")
                    rospy.sleep(0.35)
                    continue
            
                tilted = False
            
                # Case 1: We have last_seen_cy → tilt based on it
                if hasattr(self, "last_seen_cy") and self.last_seen_cy is not None:
                    img_h = self.cv_image.shape[0]
            
                    # If object was last near bottom OR very large (close)
                    if (self.last_seen_cy > img_h * 0.55) or \
                       (hasattr(self, "last_seen_area") and self.last_seen_area > target_area * 0.5):
            
                        if current_tilt is not None and current_tilt < TILT_MAX:
                            new_tilt = min(TILT_MAX, current_tilt + TILT_STEP)
                            try:
                                self.robot.camera.tilt(new_tilt)
                                self.current_tilt_cmd = float(new_tilt)
                                rospy.loginfo(f"Tilt down to {new_tilt:.2f} to reacquire target")
                            except Exception as e:
                                rospy.logwarn(f"Camera tilt failed: {e}")
                            rospy.sleep(0.3)
                            continue
                        tilted = True
            
                # Case 2: If no last_seen_cy → always try a tilt-down first
                if not tilted and current_tilt is not None and current_tilt < TILT_MAX:
                    new_tilt = min(TILT_MAX, current_tilt + TILT_STEP)
                    try:
                        self.robot.camera.tilt(new_tilt)
                        self.current_tilt_cmd = float(new_tilt)
                        rospy.loginfo(f"No reliable last seen — searching by tilting down to {new_tilt:.2f}")
                    except Exception as e:
                        rospy.logwarn(f"Camera tilt failed: {e}")
                    rospy.sleep(0.3)
                    continue
            
                # --- IF TILT DOESN’T HELP — PAN NEXT ---
                if hasattr(self, "last_seen_cx") and self.last_seen_cx is not None:
                    img_w = img.shape[1]
                    pan_step = 0.2
                    #try a quick tilt down again before panning without checking if current_tilt is none
                    

                    try:
                        if self.last_seen_cx < img_w // 2:
                            self.robot.camera.pan(-abs(pan_step))
                            rospy.loginfo("Panning left to re-acquire target")
                        else:
                            self.robot.camera.pan(abs(pan_step))
                            rospy.loginfo("Panning right to re-acquire target")
                    except Exception as e:
                        rospy.logwarn(f"Camera pan failed: {e}")
            
                    rospy.sleep(0.3)
                    continue
            
                # Last fallback
                rospy.loginfo("No last-seen position — stopping search.")
                self.cmd_pub.publish(Twist())
                break
            M = cv2.moments(largest)
            if M["m00"] == 0:
                rospy.logwarn("Zero moment, skipping frame")
                rospy.sleep(0.05)
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # record last seen centroid and area to help re-acquisition
            self.last_seen_cx = cx
            self.last_seen_cy = cy
            self.last_seen_area = area

            # Draw debug visuals
            cv2.drawContours(img, [largest], -1, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            cv2.imshow("Approach - Red Detection", img)
            cv2.imshow("Approach - Mask", mask)
            cv2.waitKey(1)

            # Compute control signals
            error_x = cx - img.shape[1] // 2
            ang = -kp_ang * error_x

            vel = Twist()
            reached = False

            # If depth is available, use averaged depth around centroid to determine distance
            if self.depth_image is not None:
                h, w = self.depth_image.shape
                sx = max(0, cx - 2)
                ex = min(w, cx + 3)
                sy = max(0, cy - 2)
                ey = min(h, cy + 3)
                window = self.depth_image[sy:ey, sx:ex]
                # compute mean ignoring NaNs
                mean_depth = float(np.nanmean(window)) if window.size > 0 else float('nan')
                
                if mean_depth > 10: #
                    mean_depth = mean_depth / 1000.0  # convert mm to m , abs forgot to check units
                else:
                    mean_depth = mean_depth  # already in meters, likely not
                if np.isfinite(mean_depth):
                    rospy.loginfo(f"Mean depth at centroid: {mean_depth:.3f} m")
                    if mean_depth <= distance_threshold:
                        rospy.loginfo("Depth threshold reached. Stopping.")
                        reached = True

            # Fallback: require area to exceed threshold for several consecutive frames
            if not reached and not np.isfinite(mean_depth):
                if area >= target_area:
                    consec_count += 1
                    rospy.loginfo(f"Area >= target ({consec_count}/{consec_needed})")
                    if consec_count >= consec_needed:
                        rospy.loginfo("Area-based consecutive threshold reached. Stopping.")
                        reached = True
                else:
                    consec_count = 0

            if reached:
                self.cmd_pub.publish(Twist())
                break

            # Simple forward controller requested: constant forward speed
            # and angular proportional to pixel error (matches user's snippet).
            vel = Twist()
            vel.linear.x = 0.15
            vel.angular.z = -0.002 * error_x
            self.cmd_pub.publish(vel)
            rospy.loginfo(f"Approaching: area={area:.0f}, cx={cx}, ang={vel.angular.z:.3f}")

            self.rate.sleep()

        # ensure robot is stopped
        self.cmd_pub.publish(Twist())

    def pickup(self):
        self.arm.go_to_home_pose()
        self.arm.set_single_joint_position("waist", 0)
        self.arm.set_ee_cartesian_trajectory(x=-0.03, z=-0.21)
        self.arm.set_ee_cartesian_trajectory(x=0.02)
        self.gripper.close(2.0)
        self.arm.set_ee_cartesian_trajectory(z=0.1)

    def drop(self):
        self.arm.set_ee_cartesian_trajectory(z=-0.1)
        self.gripper.open(2.0)
        self.arm.set_ee_cartesian_trajectory(x=-0.02)
        self.arm.go_to_sleep_pose()

    def run(self):
        self.scan_camera2()

        if self.Obj_found:
            rospy.loginfo("obj found!")
            self.approach_obj()
        else:
            rospy.loginfo("obj not found.")


if __name__ == "__main__":
    
    finder = ObjFinder()
    finder.run()
