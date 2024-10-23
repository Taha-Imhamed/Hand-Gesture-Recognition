import cv2
import mediapipe as mp
import os

# Define paths to the model files
BASE_DIR = r'C:\Users\memo\PycharmProjects\Hand Gesture Recognition System\models'
HAND_DETECTOR_MODEL = os.path.join(BASE_DIR, 'mobilenet_iter_73000.caffemodel')
DEPLOY_PROTOTXT = os.path.join(BASE_DIR, 'deploy.prototxt')

# Check if model files exist
if not os.path.isfile(DEPLOY_PROTOTXT):
    print(f"Deploy prototxt file not found: {DEPLOY_PROTOTXT}")
    exit()

if not os.path.isfile(HAND_DETECTOR_MODEL):
    print(f"Caffe model file not found: {HAND_DETECTOR_MODEL}")
    exit()

# Load the hand gesture recognition model
try:
    net = cv2.dnn.readNetFromCaffe(DEPLOY_PROTOTXT, HAND_DETECTOR_MODEL)
    print("Hand gesture model loaded successfully.")
except cv2.error as e:
    print(f"Error loading hand gesture model: {e}")
    exit()

# Initialize MediaPipe hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame with detected hands
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
