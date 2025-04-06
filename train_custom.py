import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 30

def load_custom_dataset(data_dir="Data"):  # Changed to use your existing Data folder
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Get all label folders
    labels = []
    try:
        labels = sorted([d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))])
    except FileNotFoundError:
        print(f"ERROR: No '{data_dir}' directory found. Please create it and add your label folders.")
        exit()

    if not labels:
        print(f"ERROR: No label folders found in '{data_dir}'. Please add folders with your images.")
        exit()

    print(f"\nFound {len(labels)} custom labels: {labels}\n")

    images = []
    label_ids = []
    
    for label_id, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        image_count = 0
        
        for img_name in os.listdir(label_dir):
            if img_name.endswith('.jpg') and img_name.startswith(label):
                img_path = os.path.join(label_dir, img_name)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    label_ids.append(label_id)
                    image_count += 1
                except Exception as e:
                    print(f"Warning: Could not load {img_path} - {str(e)}")
        
        print(f"Loaded {image_count} images for label '{label}'")

    if not images:
        print("ERROR: No valid images found in any label folders!")
        exit()

    X = np.array(images)
    y = utils.to_categorical(label_ids, num_classes=len(labels))
    
    return X, y, labels

def build_custom_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_custom_model():
    print("\nLoading custom dataset...")
    X, y, labels = load_custom_dataset()
    X = X / 255.0  # Normalize
    
    print("\nSplitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nBuilding model...")
    model = build_custom_model(X_train.shape[1:], len(labels))
    model.summary()
    
    print("\nTraining model...")
    model.fit(X_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              validation_data=(X_val, y_val))
    
    model.save('custom_model.h5')
    np.save('custom_labels.npy', np.array(labels))
    print(f"\nTraining complete! Model saved for labels: {labels}")

if __name__ == "__main__":
    train_custom_model()