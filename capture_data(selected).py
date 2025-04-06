import cv2
import os
from cvzone.HandTrackingModule import HandDetector
from tkinter import Tk, simpledialog

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Function to create a folder for the label
def create_label_folder(label):
    folder = os.path.join("Data", label)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

# Function to get the next available file number
def get_next_file_number(folder, label):
    existing_files = [f for f in os.listdir(folder) if f.startswith(label) and f.endswith(".jpg")]
    if not existing_files:
        return 1
    # Extract numbers from filenames and find the maximum
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
    return max(numbers) + 1

# Function to get label from a Tkinter dialog box
def get_label_from_dialog():
    root = Tk()
    root.withdraw()  # Hide the root window
    label = simpledialog.askstring("Label Input", "Enter Label (A-Z, 0-9):")
    return label

# Main loop
while True:
    # Get label from Tkinter dialog box
    label = get_label_from_dialog()
    if label is None:
        break

    # Create folder for the label
    folder = create_label_folder(label)

    # Get the next available file number
    counter = get_next_file_number(folder, label)

    print(f"Capturing images for label: {label}")
    print("Press 's' to save an image, 'd' to finish this label, and 'q' to quit.")

    while True:
        # Read frame from the camera
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        # Detect hands in the frame
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img.shape[1], x + w + padding)
            y_max = min(img.shape[0], y + h + padding)

            # Crop hand region
            imgCrop = img[y_min:y_max, x_min:x_max]

            # Check if the cropped image is not empty
            if imgCrop.size == 0:
                print("No hand detected in the frame. Skipping resize...")
            else:
                # Resize the cropped image to 200x200
                imgCrop = cv2.resize(imgCrop, (200, 200))  # Resize to 200x200
                cv2.imshow("Cropped Hand", imgCrop)

        # Display instructions on the camera feed
        cv2.putText(img, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "Press 's' to save, 'd' to finish, 'q' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Feed", img)

        key = cv2.waitKey(1)
        if key == ord("s"):  # Press 's' to save image
            if hands and imgCrop.size != 0:  # Ensure a hand is detected and imgCrop is not empty
                save_path = os.path.join(folder, f"{label}_{counter:03d}.jpg")
                cv2.imwrite(save_path, imgCrop)
                print(f"Saved: {save_path}")
                counter += 1  # Increment the counter for the next image
            else:
                print("No hand detected. Cannot save image.")

        if key == ord("d"):  # Press 'd' to finish this label
            print(f"Finished capturing images for label: {label}")
            break

        if key == ord("q"):  # Press 'q' to quit
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()