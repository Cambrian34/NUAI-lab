import cv2

# Load template
template = cv2.imread("target.png", 0)
w, h = template.shape[::-1]

cap = cv2.VideoCapture(0)

THRESHOLD = 0.65  # Adjust if needed

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Template match
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > THRESHOLD:
        # Top-left corner of match
        x, y = max_loc
        # Bottom-right corner
        br = (x + w, y + h)

        # Draw box
        cv2.rectangle(frame, (x, y), br, (0, 255, 0), 2)

        # Object center
        cx = x + w // 2
        cy = y + h // 2

        print("Detected! Confidence:", max_val)
        print("Center:", cx, cy)

        # === ROBOT LOGIC EXAMPLE ===
        mid = frame.shape[1] // 2

        if cx < mid - 50:
            print("Turn LEFT")
        elif cx > mid + 50:
            print("Turn RIGHT")
        else:
            print("Move FORWARD")

    else:
        print("Object not found")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:  # ESC = exit
        break

cap.release()
cv2.destroyAllWindows()a