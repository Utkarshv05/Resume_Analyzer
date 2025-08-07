import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing for visualization.
mp_drawing = mp.solutions.drawing_utils

# Particle class for managing particle attributes
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-10, 10)
        self.vy = random.uniform(-10, 10)
        self.size = random.randint(10, 20)
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def move(self):
        self.x += self.vx
        self.y += self.vy

        # Bounce back if particles hit the screen edge
        if self.x < 0 or self.x > width:
            self.vx = -self.vx
        if self.y < 0 or self.y > height:
            self.vy = -self.vy

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)
        #cv2.rectangle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)

# Function to check for the "victory" gesture.
def is_victory_hand(landmarks):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    index_dist = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([index_mcp.x, index_mcp.y]))
    middle_dist = np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([middle_mcp.x, middle_mcp.y]))
    ring_dist = np.linalg.norm(np.array([ring_tip.x, ring_tip.y]) - np.array([ring_mcp.x, ring_mcp.y]))

    return index_dist > 0.1 and middle_dist > 0.1 and ring_dist < 0.1

# Start video capture.
cap = cv2.VideoCapture(0)

# Get the width and height of the video feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to hold particles
particles = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_victory_hand(hand_landmarks.landmark):
                cv2.putText(frame, 'Victory!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                
                # Generate new particles if victory gesture detected
                for _ in range(5):
                    particle = Particle(random.randint(0, width), random.randint(0, height))
                    particles.append(particle)

    # Move and draw particles
    for particle in particles:
        particle.move()
        particle.draw(frame)

    # Clear particles that are out of bounds
    particles = [particle for particle in particles if 0 <= particle.x <= width and 0 <= particle.y <= height]

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
