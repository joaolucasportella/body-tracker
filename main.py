import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
pTime = 0

# Webcam input:
CAP_WIDTH = 1280
CAP_HEIGHT = 960
cap = cv2.VideoCapture(0)
cap.set(3, CAP_WIDTH)
cap.set(4, CAP_HEIGHT)

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image)
    face_results = face_detection.process(image)
    pose_results = pose.process(image)

    # Draw the body annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(image, detection)
    mp_drawing.draw_landmarks(
        image,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    fps_str = "Fps: " + str(int(fps))
    cv2.putText(image, fps_str, (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # Display
    cv2.imshow('Body Tracker 1.0', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Close window
    if cv2.getWindowProperty('Body Tracker 1.0', cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
