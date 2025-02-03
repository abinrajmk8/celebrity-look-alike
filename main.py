import cv2
import os
import warnings
import pickle  # For caching sample embeddings
import matplotlib.pyplot as plt
from deepface import DeepFace
import concurrent.futures
import gdown

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the "Samples" directory exists
os.makedirs("Samples", exist_ok=True)

# DeepFace model directory
model_dir = os.path.expanduser("~/.deepface/weights")
os.makedirs(model_dir, exist_ok=True)

# Caching file for sample embeddings
cache_file = "sample_embeddings.pkl"

# Function to check if model exists
def check_model(model_name, url):
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"[+] Model {model_name} not found. Downloading now...")
        gdown.download(url, model_path, quiet=False)
        print(f"[✔] Model {model_name} downloaded successfully!")
    else:
        print(f"[✔] Model {model_name} already exists. Skipping download.")

# Check if Facenet512 model exists
print("[+] Checking if required models exist...")
check_model("facenet512_weights.h5", "https://drive.google.com/uc?id=1sR6U5g7cpm1UZ8Oi8wbnkkEZl6SxC5pM")

# Load cached embeddings if available
if os.path.exists(cache_file):
    print("[✔] Loading cached sample embeddings...")
    with open(cache_file, "rb") as f:
        sample_embeddings = pickle.load(f)
else:
    print("[+] Preloading sample images and computing embeddings (This may take a few seconds)...")
    sample_embeddings = {}
    
    for file in os.listdir("Samples"):
        if file.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join("Samples", file)
            try:
                embeddings = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)
                if embeddings:  # Ensure it's not empty
                    sample_embeddings[img_path] = embeddings[0]["embedding"]  # Extract embedding correctly
            except Exception as e:
                print(f"Skipping {file} (No face detected)")

    # Save computed embeddings to cache
    with open(cache_file, "wb") as f:
        pickle.dump(sample_embeddings, f)

print("[✔] Sample embeddings loaded successfully!")

# Open Webcam
cap = cv2.VideoCapture(0)

def find_best_match(captured_embedding, sample_embeddings):
    """Compare the captured embedding with all sample embeddings and find the best match"""
    best_match = None
    smallest_distance = float("inf")
    best_match_path = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Compare all sample images embeddings with the captured image
        results = executor.map(
            lambda img_path: (img_path, DeepFace.verify(captured_embedding, sample_embeddings[img_path], model_name="Facenet512", enforce_detection=False)['distance']),
            sample_embeddings.keys()
        )

    # Iterate through the results and find the match with the smallest distance
    for img_path, distance in results:
        print(f"Comparing with {os.path.basename(img_path)} - Distance: {distance:.4f}")
        if distance < smallest_distance:
            smallest_distance = distance
            best_match = os.path.basename(img_path)
            best_match_path = img_path

    return best_match, best_match_path, smallest_distance

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face_roi = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list) and result:
                    emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][emotion]

                    text = f"{emotion} ({confidence:.2f})"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Emotion Detection Error: {str(e)}")

    cv2.putText(frame, "Press SPACE to capture, 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Live Face & Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  
        print("📸 Capturing image...")
        captured_image_path = "captured_face.jpg"
        cv2.imwrite(captured_image_path, face_roi)
        print("[+] Scanning for a match...")

        # Fix: Compute the embedding correctly
        captured_embedding = DeepFace.represent(captured_image_path, model_name="Facenet512", enforce_detection=False)
        captured_embedding = captured_embedding[0]["embedding"]  # Fix: Extract first embedding

        # Find the best match by comparing all sample embeddings
        best_match, best_match_path, smallest_distance = find_best_match(captured_embedding, sample_embeddings)

        if best_match:
            print(f"✅ Closest Match Found: {best_match}")
            print(f"🖼 Matched Image Path: {best_match_path}")
            print(f"🔹 Distance: {smallest_distance}")
            print(f"🔹 Model Used: Facenet512")
            print("-" * 50)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            img1 = cv2.imread(captured_image_path)
            img2 = cv2.imread(best_match_path)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            axes[0].imshow(img1)
            axes[0].axis("off")
            axes[0].set_title("Captured Face")

            axes[1].imshow(img2)
            axes[1].axis("off")
            axes[1].set_title(f"Closest Match ({best_match})")

            plt.show()

        os.remove(captured_image_path)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  
