import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detected_image):
    hand_landmark_list = detected_image.hand_landmarks
    handedness_list = detected_image.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmark_list)):
        # Loop through hands in detected hands
        hand_landmarks = hand_landmark_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10
        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    1, (88, 205, 54), 1, cv.LINE_AA)
    return annotated_image


# setup camera
cam = cv.VideoCapture(0)

# config hand detector
options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='./data/hand_landmarker.task'),
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# main loop
while True:
    ret, frame = cam.read()
    if not ret:
        print("can't receive frame from camera")
    # detect hand from frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detected_frame = detector.detect(mp_image)
    img = draw_landmarks_on_image(frame, detected_frame)
    cv.imshow("hand detection", img)
    if cv.waitKey(1) == ord("q"):
        print("Exit loop")
        break


cam.release()
cv.destroyAllWindows()
