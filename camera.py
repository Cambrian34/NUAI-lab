import cv2
cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Camera not found!")
else:
    print("Camera opened successfully.")
    cap.release()

from ultralytics import YOLO
import cv2

# Load your trained YOLOv11 model
model = YOLO("best.pt")  # <-- change path if needed

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO inference
    results = model(frame)

    # Display annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 - Real-time Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


