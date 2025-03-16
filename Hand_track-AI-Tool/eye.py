import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen size
screen_width, screen_height = pyautogui.size()

# Eye landmarks for left and right eye (MediaPipe FaceMesh indexes)
LEFT_EYE = [33, 133]  # Inner and outer corner of left eye
RIGHT_EYE = [362, 263]  # Inner and outer corner of right eye
BLINK_LANDMARKS = [159, 145, 386, 374]  # Upper and lower eyelid

def get_eye_center(landmarks, frame_width, frame_height):
    """Calculate the center of both eyes based on detected landmarks."""
    left_eye = np.mean([[landmarks[p].x * frame_width, landmarks[p].y * frame_height] for p in LEFT_EYE], axis=0)
    right_eye = np.mean([[landmarks[p].x * frame_width, landmarks[p].y * frame_height] for p in RIGHT_EYE], axis=0)
    
    # Average of both eyes gives gaze center
    return (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

def detect_blink(landmarks, frame_height):
    """Check if the user is blinking by measuring eyelid distances."""
    left_eye_height = abs((landmarks[159].y - landmarks[145].y) * frame_height)
    right_eye_height = abs((landmarks[386].y - landmarks[374].y) * frame_height)
    
    return left_eye_height < 5 and right_eye_height < 5  # If both eyes nearly closed

# Start webcam feed
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for natural movement
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Process face landmarks
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye center for cursor movement
                eye_x, eye_y = get_eye_center(face_landmarks.landmark, w, h)
                screen_x = int((eye_x / w) * screen_width)
                screen_y = int((eye_y / h) * screen_height)
                
                # Move mouse
                pyautogui.moveTo(screen_x, screen_y)
                
                # Blink detection for clicking
                if detect_blink(face_landmarks.landmark, h):
                    pyautogui.click()
        
        # Display webcam feed with landmarks
        cv2.imshow("Eye Control Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
