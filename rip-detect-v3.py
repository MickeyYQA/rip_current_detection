# rip-detect-v3.py

import numpy as np
import cv2
from djitellopy import Tello
import joblib

def load_model(model_path: str):
    """Load the pre-trained model."""
    return joblib.load(model_path)

def connect_drone() -> Tello:
    """Connect to the Tello drone and start the video stream."""
    tello = Tello()
    tello.connect()
    tello.streamon()
    return tello

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess the image for model prediction."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (80, 80)) / 255.0
    img_resized = np.asarray(img_resized * 255, dtype=np.uint8)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    return img_gray.flatten()[np.newaxis, :]

def display_prediction_status(color_state: tuple, text: str):
    """Display a window showing the prediction status with text."""
    window_name = "Prediction Status"
    window_size = (200, 100)
    status_img = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
    status_img[:] = color_state

    # Set text properties
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.6
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (status_img.shape[1] - text_size[0]) // 2
    text_y = (status_img.shape[0] + text_size[1]) // 2

    text_color = (255, 255, 255) if color_state == (0, 0, 255) else (0, 0, 0)

    cv2.putText(status_img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    cv2.imshow(window_name, status_img)

def main():
    model = load_model('model.pkl')
    tello = connect_drone()
    color_state = (0, 255, 0)
    status_text = "SAFE"
    total_predictions = 0
    confidence = 0
    try:
        while True:
            #(omitted some code)
            img = tello.get_frame_read().frame
            img_preprocessed = preprocess_image(img)

            cv2.imshow("Tello Video Stream", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prediction = model.predict(img_preprocessed)
            total_predictions += 1
            # confidence : times of prediction that shows rip current exists
            if prediction[0] == 1:
                confidence += 1
                print("Exists rip current")
            else:
                print("No rip current")
    
            # once reaches 100 times of prediction
            if total_predictions >= 100:
                if confidence > 82: 
                    # 82 is an empirical confidence threshold where accuracy is maximized
                    color_state = (0, 0, 255)  # red
                    status_text = "RIP CURRENT"
                else:
                    color_state = (0, 255, 0)  # green
                    status_text = "SAFE"
        
                # Reset counter for next round of prediction.
                total_predictions = 0
                confidence = 0

            display_prediction_status(color_state, status_text)
    finally:
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
