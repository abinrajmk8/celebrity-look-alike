# Celebrity Look-Alike Finder

This project identifies the closest celebrity match for a given input image using DeepFace for facial recognition. The system compares a given image to a database of celebrity images and returns the closest match along with additional emotion recognition. This uses the **DeepFace** library for facial recognition, model loading, and embedding comparison.

---

## 📌 Prerequisites  

Before setting up the project, ensure you have the following:

- **Python 3.10** or later  
- **pip 25.0** or later  
- **Virtual Environment** (recommended)

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository  
First, clone the repository to your local machine. Use the link you will provide for the `Samples` folder later.

```bash
git clone <repository-link>
cd <project-directory>
```

---

### 2️⃣ Create & Activate a Virtual Environment  
It’s recommended to use a virtual environment to manage dependencies for this project. You can create and activate the virtual environment using the following commands:

#### For macOS/Linux:  
```bash
python -m venv venv
source venv/bin/activate
```

#### For Windows:  
```bash
python -m venv project_env
project_env\Scripts\activate
```

> Note: Replace `project_env` with the name of your virtual environment.

---

### 3️⃣ Install Required Packages  
After activating the virtual environment, install the required packages for the project by running the following command:

```bash
pip install -r requirements.txt
```

> Tip: After adding or updating any packages, make sure to update `requirements.txt` using the following command:

```bash
pip freeze > requirements.txt
```

---

### 4️⃣ Set Up the Samples Folder  
Before you run the project, ensure that the `Samples` folder is populated with images of celebrities. You can clone or download the folder using the link you will provide. The system will load the images from this folder, compute embeddings, and store them in `sample_embeddings.pkl`.

---

### 5️⃣ Run the Project  
Now that the setup is complete, run the project using the following command:

```bash
python celebrityDetector.py
```

> First Run: The first run of the program will take some time, as it will:
> - Download the required pre-trained models.
> - Process the images in the `Samples` folder.
> - Compute embeddings for each image.
> - Store the embeddings in `sample_embeddings.pkl` for faster future matches.

This initial run can take **5-10 minutes**, depending on the number of images in the `Samples` folder and the system’s performance.

---

### 6️⃣ Modify Cache for New or Deleted Images  
To ensure your cache is always up to date, you can modify it whenever new images are added or existing images are deleted. You can do this using the `preload.py` script:

```bash
python preload.py
```

This will:
- Fully preload data (use Option 1) or
- Modify the cache to reflect any added or deleted images (use Option 2).

---

## 💡 How to Use

Once the program is set up and the data is preloaded, you can run the main program (`celebrityDetector.py`), and it will:
- Capture a face from the webcam.
- Compare the captured image with the cached images.
- Show the closest match and the corresponding distance.
- Display the captured face and its closest match side by side.

---

## 📝 Summary

This project uses **DeepFace** for facial recognition and matching celebrity images to a given input. The project is designed to work by preloading and caching celebrity images and embeddings. The steps include:
- Setting up a virtual environment.
- Installing the required packages.
- Preloading data for the first time.
- Running the program to capture and compare faces.
- Updating the cache when new images are added or deleted.

With these steps, you can run the **Celebrity Look-Alike Finder** and enjoy identifying the closest celebrity match based on facial recognition.
