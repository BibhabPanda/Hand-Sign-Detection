import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

class CustomGestureDetector:
    def __init__(self, model_path='custom_model.h5'):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.labels = np.load('custom_labels.npy', allow_pickle=True)
            self.detector = HandDetector(maxHands=1, detectionCon=0.8)
            self.img_size = 200
            print(f"\nLoaded custom gesture detector for: {list(self.labels)}")
            print("Press 'Q' to quit detection\n")
        except Exception as e:
            print(f"\nERROR: Could not load model - {str(e)}")
            print("Please train the model first using train_custom.py")
            exit()

    def preprocess(self, img_crop):
        img = cv2.resize(img_crop, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img / 255.0).reshape(1, self.img_size, self.img_size, 3)

    def predict(self, img):
        hands = self.detector.findHands(img, draw=False)[0]
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Add 30% padding
            padding = int(max(w, h) * 0.3)
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img.shape[1], x + w + padding)
            y_max = min(img.shape[0], y + h + padding)
            
            img_crop = img[y_min:y_max, x_min:x_max]
            
            if img_crop.size > 0:
                # Show what the model sees
                debug_img = cv2.resize(img_crop, (300, 300))
                cv2.imshow("Model Input Preview", debug_img)
                
                processed = self.preprocess(img_crop)
                pred = self.model.predict(processed, verbose=0)[0]
                top_idx = np.argmax(pred)
                confidence = pred[top_idx]
                
                if confidence > 0.85:  # Confidence threshold
                    label = self.labels[top_idx]
                    cv2.putText(img, f"{label} ({confidence:.2f})", 
                              (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0,255,0), 2)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                    return label, img
        
        return None, img

def main():
    detector = CustomGestureDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
            
        img = cv2.flip(img, 1)  # Mirror effect
        _, img = detector.predict(img)
        cv2.imshow("Custom Gesture Detection", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection ended")

if __name__ == "__main__":
    main()