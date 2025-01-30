import cv2
import mediapipe as mp
import sys
import os

def crop_face_mediapipe(image, output_folder="output", padding=20):
    """
    Detects and crops all faces found in an image using Mediapipe.
    
    :param image: OpenCV image (numpy array)
    :param output_folder: Folder to save cropped face images
    :param padding: Extra pixels around each face (default: 20)
    :return: List of cropped face images (numpy arrays)
    """
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    # Convert to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Detector
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_image)

        # If no faces detected, return empty list
        if not results.detections:
            print("No faces detected.")
            return []

        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cropped_faces = []
        ih, iw, _ = image.shape

        # Process all detected faces
        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * iw) - padding
            y = int(bboxC.ymin * ih) - padding
            w = int(bboxC.width * iw) + 4 * padding
            h = int(bboxC.height * ih) + 4 * padding

            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            # Crop the face
            cropped_face = image[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)

            # Save each cropped face
            output_path = os.path.join(output_folder, f"cropped_face_{i+1}.jpg")
            cv2.imwrite(output_path, cropped_face)
            print(f"Cropped face {i+1} saved to {output_path}")

        return cropped_faces  # Return list of cropped face images
