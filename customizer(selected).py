import os
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def get_next_number(label_dir, label):
    """Find the highest existing number in a label directory."""
    files = glob.glob(os.path.join(label_dir, f"{label}_[0-9][0-9][0-9].jpg"))
    if not files:
        return 1
    numbers = [int(f.split("_")[-1].split(".")[0]) for f in files]
    return max(numbers) + 1

def validate_image(image):
    """Check if image has valid dimensions and channels."""
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Invalid image shape: {image.shape}")
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] not in (1, 3):
        raise ValueError(f"Invalid channel count: {image.shape[-1]}")
    return image

def generate_augmented_images(label_dir, label, num_augmented_images=5):
    """Generate augmented images with continuous numbering."""
    starting_number = get_next_number(label_dir, label)
    original_images = sorted(glob.glob(os.path.join(label_dir, f"{label}_[0-9][0-9][0-9].jpg")))
    
    if not original_images:
        print(f"No original images found for label '{label}'. Skipping...")
        return
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.9, 1.1],
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    for img_path in original_images:
        try:
            # Load and validate image
            image = Image.open(img_path)
            image = np.array(image)
            image = validate_image(image)
            
            # Reshape for augmentation (1, height, width, channels)
            image = image.reshape((1,) + image.shape)
            
            # Generate augmented versions
            for i, batch in enumerate(datagen.flow(image, batch_size=1)):
                if i >= num_augmented_images:
                    break
                
                augmented = Image.fromarray(batch[0].astype('uint8').squeeze())
                filename = f"{label}_{starting_number:03d}.jpg"
                save_path = os.path.join(label_dir, filename)
                augmented.save(save_path)
                print(f"Saved: {save_path}")
                starting_number += 1
                
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {str(e)}")

if __name__ == "__main__":
    data_dir = "Data"
    num_augmented_images = 100
    labels = [chr(i) for i in range(ord('A'), ord('Z')+1)] + [str(i) for i in range(10)]

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        
        if not os.path.exists(label_dir):
            print(f"Directory not found for label '{label}'. Skipping...")
            continue

        print(f"\nProcessing label '{label}'")
        generate_augmented_images(label_dir, label, num_augmented_images)

    print("\nAugmentation complete!")