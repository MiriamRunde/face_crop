import requests
import cv2
import numpy as np
import sys

def download_and_load_function(github_raw_url):
    """Downloads the face_crop.py script from GitHub and loads the function dynamically."""
    response = requests.get(github_raw_url)
    if response.status_code != 200:
        print(f"Error: Could not fetch script from {github_raw_url}")
        sys.exit(1)

    script_content = response.text
    exec(script_content, globals())  # Execute script in the global namespace

    if "crop_face_mediapipe" not in globals():
        print("Error: Function 'crop_face_mediapipe' not found in the script.")
        sys.exit(1)

    print("Function loaded successfully.")

def download_image(image_url):
    """Downloads an image from a URL and converts it to an OpenCV image."""
    response = requests.get(image_url)
    if response.status_code != 200:
        print(f"Error: Could not download image from {image_url}")
        sys.exit(1)

    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

if __name__ == "__main__":
    # GitHub raw link to your face_crop.py
    github_script_url = "https://raw.githubusercontent.com/MiriamRunde/face_crop/main/crop.py"  # Adjust if needed

    # Image URL (or replace with a local file path)
    image_url = "https://raw.githubusercontent.com/MiriamRunde/face_crop/main/data/sc3.png"

    # Output path
    output_image_path = "cropped_face.jpg"

    # Load the function from GitHub
    download_and_load_function(github_script_url)

    # Download the image
    image = download_image(image_url)

    # Call the function
    cropped_face = crop_face_mediapipe(image)

    if cropped_face is not None:
        cv2.imwrite(output_image_path, cropped_face)
        print(f"Cropped face saved to {output_image_path}")
    else:
        print("No face detected in the image.")
