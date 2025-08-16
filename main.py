import os
import cv2
import numpy as np
from HandTrackingModule import HandDetector

# Initialize hand detector (detects 1 hand, with minimum confidence 0.85)
hand_detector = HandDetector(max_num_hands=1, min_detection_confidence=0.85)

# Open webcam
webcam = cv2.VideoCapture(0)

# Set webcam properties: resolution + FPS
webcam.set(propId=3, value=1280)  # Width
webcam.set(propId=4, value=720)  # Height
webcam.set(propId=cv2.CAP_PROP_FPS, value=60)  # Frames per second

# Load header images (color selection templates) from "paint_templates" folder
files_list = os.listdir(path="paint_templates")
headers = []
for file in files_list:
    header = cv2.imread(filename=f"paint_templates/{file}")
    headers.append(header)

# Default header and drawing color
header = headers[0]
draw_color = (255, 0, 255)

# Previous finger positions (used for drawing lines)
xp, yp = 0, 0

# Brush and eraser sizes
brush_size = 15
eraser_size = 40

# Empty canvas where drawing will be stored
blank_frame = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)


# -----------------------------
# Main Loop
# -----------------------------
while True:
    is_successful, frame = webcam.read()
    if not is_successful:
        break

    # Detect hands
    hands = hand_detector.findHands(frame)

    if len(hands) > 0:
        hand = hands[0]
        landmark_list = hand["landmark list"]

        # Index finger tip (x1, y1), Middle finger tip (x2, y2)
        x1, y1 = landmark_list[8]["x"], landmark_list[8]["y"]
        x2, y2 = landmark_list[12]["x"], landmark_list[12]["y"]

        # Check which fingers are up
        fingers = hand_detector.fingersUp(hand_no=0)

        # ---------------- Selection Mode (2 fingers up) ----------------
        if fingers["index"] == 1 and fingers["middle"] == 1:

            # If fingers are in the header region → change tool/color
            if y1 < 120:
                if x1 > 200 and x1 < 400:  # Purple/Pink
                    header = headers[0]
                    draw_color = (255, 0, 255)

                elif x1 > 500 and x1 < 700:  # Blue
                    header = headers[1]
                    draw_color = (255, 0, 0)

                elif x1 > 780 and x1 < 950:  # Green
                    header = headers[2]
                    draw_color = (0, 210, 0)

                elif x1 > 1040 and x1 < 1280:  # Eraser
                    header = headers[3]
                    draw_color = (0, 0, 0)

            # Draw selection rectangle on screen
            cv2.rectangle(
                img=frame,
                pt1=(x1 - 5, y1),
                pt2=(x2 + 5, y2),
                color=draw_color,
                thickness=cv2.FILLED,
            )

            # Reset previous points
            xp, yp = 0, 0

        # ---------------- Drawing Mode (only index finger up) ----------------
        if fingers["index"] == 1 and fingers["middle"] == 0:

            # Visualize drawing point
            cv2.circle(
                img=frame,
                center=(x1, y1),
                radius=brush_size,
                color=draw_color,
                thickness=cv2.FILLED,
            )

            # Initialize previous point if not set
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # If eraser is selected
            if draw_color == (0, 0, 0):
                # Show eraser circle
                cv2.circle(
                    img=frame,
                    center=(x1, y1),
                    radius=eraser_size,
                    color=draw_color,
                    thickness=cv2.FILLED,
                )

                # Erase on the blank frame
                cv2.line(
                    img=blank_frame,
                    pt1=(xp, yp),
                    pt2=(x1, y1),
                    color=draw_color,
                    thickness=eraser_size + 40,
                )

            else:
                # Draw paint on the blank frame
                cv2.line(
                    img=blank_frame,
                    pt1=(xp, yp),
                    pt2=(x1, y1),
                    color=draw_color,
                    thickness=brush_size,
                )

            # Update previous point
            xp, yp = x1, y1

    else:
        # Reset previous points when no hand is detected
        xp, yp = 0, 0

    # ---------------- Frame Merging ----------------
    # Convert blank_frame to grayscale for masking
    gray_blank_frame = cv2.cvtColor(src=blank_frame, code=cv2.COLOR_BGR2GRAY)

    # Create inverse mask
    ret_val, thresh_inv = cv2.threshold(
        src=gray_blank_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY_INV
    )
    thresh_inv = cv2.cvtColor(src=thresh_inv, code=cv2.COLOR_GRAY2BGR)

    # Combine live feed and drawing
    output_frame = cv2.bitwise_and(src1=frame, src2=thresh_inv)
    output_frame = cv2.bitwise_or(src1=output_frame, src2=blank_frame)

    # Overlay header on top
    output_frame[0:120, 0:1280] = header

    # Show final output
    cv2.imshow(winname="AI Virtual Paint", mat=output_frame)

    # ---------------- Keyboard Controls ----------------
    key = cv2.waitKey(delay=1)
    if key == 27:  # ESC → Quit
        break
    elif key == ord("c") or key == ord("C"):  # 'c' or 'C' → Clear canvas
        blank_frame = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)

# Release resources
webcam.release()
cv2.destroyAllWindows()
