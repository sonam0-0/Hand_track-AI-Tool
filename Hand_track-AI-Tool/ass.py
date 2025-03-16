import cv2
import mediapipe as mp
import pyautogui
import random
from pynput.mouse import Button, Controller

mouse = Controller()
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Utility Functions
def get_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_angle(a, b, c):
    """Calculate the angle between three points"""
    import math
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
    return math.degrees(math.acos(max(-1, min(1, cosine_angle))))  # Clamp value

def find_finger_tip(processed):
    """Extract index finger tip position from hand landmarks"""
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    """Move mouse cursor based on finger position"""
    if index_finger_tip:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)  # Fixed Y mapping
        pyautogui.moveTo(x, y)

def is_left_click(landmark_list, thumb_index_dist):
    return get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and \
           get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and \
           thumb_index_dist > 50

def is_right_click(landmark_list, thumb_index_dist):
    return get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and \
           get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and \
           thumb_index_dist > 50

def is_double_click(landmark_list, thumb_index_dist):
    return get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and \
           get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and \
           thumb_index_dist > 50

def is_screenshot(landmark_list, thumb_index_dist):
    return get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and \
           get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and \
           thumb_index_dist < 50

def detect_gesture(frame, landmark_list, processed):
    """Detect hand gestures and perform actions"""
    if len(landmark_list) < 21:  # Ensure list is long enough
        return

    index_finger_tip = find_finger_tip(processed)
    thumb_index_dist = get_distance(landmark_list[4], landmark_list[5])

    if thumb_index_dist < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
        move_mouse(index_finger_tip)
    elif is_left_click(landmark_list, thumb_index_dist):
        mouse.press(Button.left)
        mouse.release(Button.left)
        cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif is_right_click(landmark_list, thumb_index_dist):
        mouse.press(Button.right)
        mouse.release(Button.right)
        cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif is_double_click(landmark_list, thumb_index_dist):
        pyautogui.doubleClick()
        cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    elif is_screenshot(landmark_list, thumb_index_dist):
        im1 = pyautogui.screenshot()
        label = random.randint(1, 1000)
        im1.save(f'my_screenshot_{label}.png')
        cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def main():
    """Main function to capture video and process hand gestures"""
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Hand Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





