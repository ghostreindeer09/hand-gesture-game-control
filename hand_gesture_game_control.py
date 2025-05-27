import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Variables to track the hand's horizontal and vertical position to detect swipe
prev_x = None
prev_y = None
swipe_threshold_x = 40  # minimum pixel movement to consider a horizontal swipe
swipe_threshold_y = 40  # minimum pixel movement to consider a vertical swipe
last_action_time = 0
action_cooldown = 1.0  # seconds to wait between key presses to avoid spam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror image for natural movement
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  # just take the first detected hand
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the x and y coordinates of the wrist landmark (landmark 0)
        wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
        wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

        current_time = time.time()
        if prev_x is not None and prev_y is not None:
            diff_x = wrist_x - prev_x
            diff_y = wrist_y - prev_y

            # Check horizontal swipe
            if abs(diff_x) > swipe_threshold_x and (current_time - last_action_time) > action_cooldown:
                if diff_x > 0:
                    print("Swipe Right detected! Pressing right arrow key.")
                    pyautogui.press('right')
                else:
                    print("Swipe Left detected! Pressing left arrow key.")
                    pyautogui.press('left')

                last_action_time = current_time

            # Check vertical swipe
            elif abs(diff_y) > swipe_threshold_y and (current_time - last_action_time) > action_cooldown:
                if diff_y > 0:
                    print("Swipe Down detected! Pressing down arrow key.")
                    pyautogui.press('down')
                else:
                    print("Swipe Up detected! Pressing up arrow key.")
                    pyautogui.press('up')

                last_action_time = current_time

        prev_x = wrist_x
        prev_y = wrist_y

    cv2.imshow('Hand Swipe Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
