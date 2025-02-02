
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

import shutil
import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace

# Ensure samples directory exists
os.makedirs("samples", exist_ok=True)

# Dataset directory
data_dir = "Celebrity Faces Dataset"
input_image = "image1.png"  # Variable for the input image

# Copy first image from each celebrity folder to the samples folder
for directory in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, directory)
    if os.path.isdir(dir_path):  # Ensure it's a directory
        files = os.listdir(dir_path)
        if files:
            first_file = files[0]
            src = os.path.join(dir_path, first_file)
            dest = os.path.join("samples", f"{directory}.jpg")
            shutil.copyfile(src, dest)

# Ensure input image exists before proceeding
if not os.path.exists(input_image):
    raise FileNotFoundError(f"Error: {input_image} is missing!")

# Variables to track the best match
best_match = None
smallest_distance = float("inf")
best_match_path = None

# Perform face verification for each sample
for file in os.listdir("samples"):
    if file.endswith(".jpg"):
        img_path = os.path.join("samples", file)

        # Use enforce_detection=False to avoid crashes if no face is detected
        result = DeepFace.verify(input_image, img_path, enforce_detection=False)

        # Track the closest match (smallest distance)
        if result["distance"] < smallest_distance:
            smallest_distance = result["distance"]
            best_match = file
            best_match_path = img_path

# Always print the closest match, even if not verified
if best_match:
    print(f"✅ Closest Match Found: {best_match}")
    print(f"🖼 Matched Image Path: {best_match_path}")
    print(f"🔹 Distance: {smallest_distance}")
    print(f"🔹 Model Used: {result['model']}")
    print("-" * 50)  # Separator for readability

    # Display both images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Load images using OpenCV
    img1 = cv2.imread(input_image)
    img2 = cv2.imread(best_match_path)

    # Convert from BGR to RGB for proper display
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Show input image
    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title(f"Input Image ({input_image})")

    # Show best match image
    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title(f"Closest Match ({best_match})")

    # Show the images
    plt.show()
