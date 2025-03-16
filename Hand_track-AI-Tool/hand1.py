import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)
    frame_height, frame_width, _ = img.shape

    # Convert to RGB for Mediapipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hands_model.process(rgb_img)
    hands = output.multi_hand_landmarks

    x1 = y1 = x2 = y2 = 0  # Reset for each frame

    if hands:
        for hand in hands:
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index Finger Tip
                    cv2.circle(img, (x, y), 8, (0, 255, 255), 3)
                    x1, y1 = x, y

                if id == 4:  # Thumb Tip
                    cv2.circle(img, (x, y), 8, (0, 0, 255), 3)
                    x2, y2 = x, y

        # Ensure we have valid points before calculating distance
        if x1 and x2:
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 4

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

            if dist > 15:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    # Display the Output
    cv2.imshow("Hand Volume Control", img)

    # Exit on ESC key press
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

