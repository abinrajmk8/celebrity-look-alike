import os
import pickle
from deepface import DeepFace

def load_existing_names(file_path):
    """Load the existing celebrity names from a text file."""
    if not os.path.exists(file_path):
        return set()  # Return an empty set if the file doesn't exist
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file)

def save_names(names, file_path):
    """Save the updated celebrity names to the text file."""
    with open(file_path, 'w') as file:
        for name in sorted(names):  # Sort names in ascending order before saving
            file.write(name + '\n')

def load_embeddings(cache_file):
    """Load existing sample embeddings from the cache."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(sample_embeddings, cache_file):
    """Save embeddings to the cache file."""
    with open(cache_file, 'wb') as f:
        pickle.dump(sample_embeddings, f)

def compare_and_update_images(image_dir, existing_names, sample_embeddings):
    """Compare current images with the existing names and update accordingly."""
    current_images = set(os.listdir(image_dir))
    current_names = {os.path.splitext(image)[0] for image in current_images}

    # Identify missing and new images
    missing_names = existing_names - current_names
    new_names = current_names - existing_names

    # Remove missing images from the embeddings cache
    for name in missing_names:
        if name in sample_embeddings:
            del sample_embeddings[name]

    # Add new images to the embeddings cache
    for name in new_names:
        img_path = os.path.join(image_dir, name + ".jpg")  # Assuming images have .jpg extension
        try:
            embeddings = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)
            if embeddings:
                sample_embeddings[img_path] = embeddings[0]["embedding"]
        except Exception as e:
            print(f"Skipping {name} (No face detected or error occurred): {e}")

    return missing_names, new_names, sample_embeddings, current_names

def preload_data(image_dir, names_file_path, cache_file):
    """Fully preload the data (create or update the cache)."""
    existing_names = {os.path.splitext(image)[0] for image in os.listdir(image_dir)}
    sample_embeddings = {}
    
    for name in existing_names:
        img_path = os.path.join(image_dir, name + ".jpg")  # Assuming images have .jpg extension
        try:
            embeddings = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)
            if embeddings:
                sample_embeddings[img_path] = embeddings[0]["embedding"]
        except Exception as e:
            print(f"Skipping {name} (No face detected or error occurred): {e}")
    
    # Save the celebrity names and embeddings
    save_names(existing_names, names_file_path)
    save_embeddings(sample_embeddings, cache_file)
    print(f"[+] Data preloaded with {len(existing_names)} entries and embeddings saved.")

def update_cache(image_dir, names_file_path, cache_file):
    """Modify the cache to reflect any added or deleted images."""
    existing_names = load_existing_names(names_file_path)
    sample_embeddings = load_embeddings(cache_file)

    # Compare current images with the existing names in the record file
    missing_names, new_names, updated_embeddings, current_names = compare_and_update_images(image_dir, existing_names, sample_embeddings)

    if missing_names or new_names:
        # Update the names list by adding new names and removing missing ones
        existing_names.update(new_names)
        existing_names -= missing_names
        save_names(existing_names, names_file_path)

        # Update the embeddings cache
        save_embeddings(updated_embeddings, cache_file)
        
        print(f"[+] Cache updated: {len(new_names)} new names added, {len(missing_names)} names removed.")
    else:
        print("[+] No changes detected. Cache remains up to date.")

def show_menu():
    """Show menu for the user to choose actions."""
    print("\nSelect an option:")
    print("1: Preload Data")
    print("2: Modify Cache (Update for added/deleted images)")
    print("3: Exit")
    choice = input("Enter your choice: ").strip()
    return choice

if __name__ == "__main__":
    # Set your directory paths
    image_directory = 'C:\\Users\\ABINRAJ\\OneDrive\\1\\Project\\deepface\\Samples'  # Samples folder
    names_file_path = 'celebrityRecord.txt'  # Text file containing celebrity names
    cache_file = 'sample_embeddings.pkl'  # Cache file containing embeddings

    # Show menu to user and perform actions based on input
    while True:
        choice = show_menu()
        if choice == '1':
            # Fully preload data (create or update the cache)
            print("[+] Preloading data...")
            preload_data(image_directory, names_file_path, cache_file)

        elif choice == '2':
            # Modify cache to reflect added or deleted images
            print("[+] Modifying cache...")
            update_cache(image_directory, names_file_path, cache_file)

        elif choice == '3':
            print("[+] Exiting program.")
            break

        else:
            print("[!] Invalid choice. Please select a valid option.")
