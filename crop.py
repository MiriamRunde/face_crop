import cv2
import mediapipe as mp
import sys

def crop_face_mediapipe(image_path, output_path, padding=20):
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        sys.exit(1)
    
    # Convert to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Detector
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_image)

        # If no faces detected
        if not results.detections:
            print("No face detected.")
            sys.exit(1)

        # Process the first detected face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw) - padding
            y = int(bboxC.ymin * ih) - padding
            w = int(bboxC.width * iw) + 4 * padding
            h = int(bboxC.height * ih) + 4* padding

            # Ensure the coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            # Crop the face
            cropped_face = image[y:y+h, x:x+w]
            break

    # Save the cropped face
    cv2.imwrite(output_path, cropped_face)
    print(f"Cropped face saved to {output_path}")


if __name__ == "__main__":
    input_image = "data/screenshot"   # Change to your input image path
    output_image = "output/cropped_face.jpg"  # Change to desired output path
    crop_face_mediapipe(input_image, output_image)
