import cv2
import mediapipe as mp
import sys

def crop_face_mediapipe(image, padding=20):
    """
    Detects and crops the first face found in an image using Mediapipe.
    
    :param image: OpenCV image (numpy array)
    :param padding: Extra pixels around the face (default: 20)
    :return: Cropped face image (numpy array) or None if no face detected
    """
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    # Convert to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Detector
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_image)

        # If no faces detected, return None
        if not results.detections:
            print("No face detected.")
            return None

        # Process the first detected face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw) - padding
            y = int(bboxC.ymin * ih) - padding
            w = int(bboxC.width * iw) + 4 * padding
            h = int(bboxC.height * ih) + 4 * padding

            # Ensure the coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            # Crop the face
            cropped_face = image[y:y+h, x:x+w]
            return cropped_face  # Return the cropped face as an OpenCV image

    return None  # Return None if no face was processed
