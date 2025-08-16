import cv2
from mediapipe.python.solutions import hands, drawing_utils


class HandDetector:
    """A helper class for detecting and tracking hands using MediaPipe."""

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        """
        Initialize the HandDetector.

        Args:
            static_image_mode (bool): Whether to treat input images as static.
            max_num_hands (int): Maximum number of hands to detect.
            model_complexity (int): Complexity of the landmark model (0 or 1).
            min_detection_confidence (float): Minimum confidence for detection.
            min_tracking_confidence (float): Minimum confidence for tracking.
        """

        # Store configuration parameters
        self.mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model = model_complexity
        self.detection_thresh = min_detection_confidence
        self.tracking_thresh = min_tracking_confidence

        # Initialize the MediaPipe Hands model
        self.hand_detector = hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model,
            min_detection_confidence=self.detection_thresh,
            min_tracking_confidence=self.tracking_thresh,
        )

        # Landmark indices for fingertips (Index, Middle, Ring, Pinky)
        self.tip_ids = [8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        """
        Detect hands and return detailed landmark and bounding box data.

        Args:
            frame (numpy.ndarray): The BGR image frame.
            draw (bool): Whether to draw hand landmarks on the frame.

        Returns:
            list: List of dictionaries with hand information.
        """
        self.hands = []

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection
        self.results = self.hand_detector.process(image=rgb_frame)
        self.detections = self.results.multi_hand_landmarks

        # If hands are detected
        if self.detections:
            for detection, hand_type in zip(
                self.detections, self.results.multi_handedness
            ):
                landmark_list = []
                x_list = []
                y_list = []
                bounding_box = []

                # Draw landmarks and connections if enabled
                if draw:
                    drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=detection,
                        connections=hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_utils.DrawingSpec(
                            color=(0, 0, 255), thickness=cv2.FILLED, circle_radius=4
                        ),
                        connection_drawing_spec=drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2
                        ),
                    )

                # Extract landmark positions in pixel coordinates
                for position in detection.landmark:
                    x, y, z = (
                        int(position.x * frame_width),
                        int(position.y * frame_height),
                        int(position.z * frame_width),
                    )
                    x_list.append(x)
                    y_list.append(y)
                    landmark_list.append({"x": x, "y": y, "z": z})

                # Determine bounding box coordinates
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                x1, y1, x2, y2 = x_min - 20, y_min - 20, x_max + 20, y_max + 20
                width, height = x2 - x1, y2 - y1

                bounding_box = {"x": x1, "y": y1, "width": width, "height": height}

                # Add center of hand as an extra landmark
                landmark_list.append({"x": x1 + width // 2, "y": y1 + height // 2})

                # Save all hand data
                self.hands.append(
                    {
                        "landmark list": landmark_list,
                        "bounding box": bounding_box,
                        "hand type": hand_type.classification[0].label,
                    }
                )

        return self.hands

    def fingersUp(self, hand_no):
        """
        Determine which fingers are raised.

        Args:
            hand_no (int): Index of the detected hand.

        Returns:
            dict: Finger status (1 = up, 0 = down).
        """
        hand = self.hands[hand_no]
        finger_names = ["index", "middle", "ring", "pinky"]
        fingers = {}

        if self.detections:
            # Thumb detection logic (different for left/right hand)
            if hand["hand type"] == "Right":
                if hand["landmark list"][4]["x"] < hand["landmark list"][3]["x"]:
                    fingers["thumb"] = 1
                else:
                    fingers["thumb"] = 0
            else:
                if hand["landmark list"][4]["x"] > hand["landmark list"][3]["x"]:
                    fingers["thumb"] = 1
                else:
                    fingers["thumb"] = 0

            # Check other fingers
            for i, tip_id in enumerate(self.tip_ids):
                finger_name = finger_names[i]
                if (
                    hand["landmark list"][tip_id]["y"]
                    < hand["landmark list"][tip_id - 2]["y"]
                ):
                    fingers[finger_name] = 1
                else:
                    fingers[finger_name] = 0

        return fingers

    def findDistance(self, frame, hand_no1, hand_no2, id1, id2, draw=True):
        """
        Calculate distance between two landmarks from two different hands.

        Args:
            frame (numpy.ndarray): The BGR image frame.
            hand_no1 (int): First hand index.
            hand_no2 (int): Second hand index.
            id1 (int): Landmark ID from the first hand.
            id2 (int): Landmark ID from the second hand.
            draw (bool): Whether to draw visual aids.

        Returns:
            dict: Distance and coordinates between landmarks.
        """
        hand1 = self.hands[hand_no1]
        hand2 = self.hands[hand_no2]

        x1, y1 = hand1["landmark list"][id1]["x"], hand1["landmark list"][id1]["y"]
        x2, y2 = hand2["landmark list"][id2]["x"], hand2["landmark list"][id2]["y"]

        # Euclidean distance
        distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

        # Midpoint between the two landmarks
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw on frame if enabled
        if draw:
            cv2.circle(
                img=frame,
                center=(x1, y1),
                radius=15,
                color=(255, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.circle(
                img=frame,
                center=(x2, y2),
                radius=15,
                color=(255, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.line(
                img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3
            )
            cv2.circle(
                img=frame,
                center=(cx, cy),
                radius=15,
                color=(210, 0, 210),
                thickness=cv2.FILLED,
            )

        return {
            "distance": distance,
            "id1x": x1,
            "id1y": y1,
            "id2x": x2,
            "id2y": y2,
            "center x": cx,
            "center y": cy,
        }


def main():
    """Run a simple webcam test for hand tracking."""
    webcam = cv2.VideoCapture(1)
    hand_detector = HandDetector()

    while True:
        is_successful, frame = webcam.read()
        if not is_successful:
            break

        # Detect and draw hands
        hand_detector.findHands(frame)

        # Display the frame
        cv2.imshow(winname="frame", mat=frame)

        # Exit on ESC key
        key = cv2.waitKey(delay=1)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
