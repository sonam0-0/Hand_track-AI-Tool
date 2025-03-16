import cv2
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Variables for 3D rendering
cube_position = [0.0, 0.0, -5.0]  # Cube's initial position

# Function to draw a cube
def draw_cube():
    glBegin(GL_QUADS)
    
    # Define the vertices of the cube
    vertices = [
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1),
        (-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1),
        (-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1),
        (-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)
    ]

    # Draw the cube faces
    for vertex in vertices:
        glVertex3fv(vertex)
    
    glEnd()

# Initialize OpenGL
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

# Set up the 3D viewport
def setup_viewport():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

# Function to update the cube's position based on hand movement
def update_cube_position(hand_landmarks, frame_width, frame_height):
    global cube_position

    # Get the coordinates of the wrist (landmark 0) and index finger tip (landmark 8)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Convert normalized coordinates to screen coordinates
    wrist_coords = (int(wrist.x * frame_width), int(wrist.y * frame_height))
    index_coords = (int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height))

    # Calculate the movement in X and Y direction
    cube_position[0] = (index_coords[0] - wrist_coords[0]) / 50.0  # X-axis movement
    cube_position[1] = -(index_coords[1] - wrist_coords[1]) / 50.0  # Y-axis movement

# Main function to run the program
def main():
    cap = cv2.VideoCapture(0)  # Start video capture

    # Set up OpenGL rendering
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("3D Cube with Hand Tracking")

    init()
    setup_viewport()

    # Function to render the OpenGL scene
    def render_scene():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Move the cube based on hand movement
        glTranslatef(cube_position[0], cube_position[1], cube_position[2])
        draw_cube()

        glutSwapBuffers()

    # Display and idle functions for OpenGL
    glutDisplayFunc(render_scene)
    glutIdleFunc(render_scene)

    # Start capturing video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Update the 3D cube position based on hand landmarks
                update_cube_position(landmarks, frame.shape[1], frame.shape[0])

        # Display the frame with hand landmarks
        cv2.imshow("Hand Tracking", frame)

        # Break the loop on pressing 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture and close OpenGL window
    cap.release()
    cv2.destroyAllWindows()

    # Start the OpenGL main loop
    glutMainLoop()

# Run the program
if __name__ == "__main__":
    main()
