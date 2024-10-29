import time
import cv2
from datetime import datetime
import os

def capture_images(folder_path):
    # Open a connection to the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if ret:
                # Generate a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"{timestamp}.jpg"
                file_path = os.path.join(folder_path, filename)
                
                # Save the image to the specified folder
                cv2.imwrite(file_path, frame)
                print(f"Image saved: {file_path}")

                time.sleep(0.25)
            else:
                print("Error: Could not capture image.")
    except KeyboardInterrupt:
        pass
    
    # Release the camera
    cap.release()

capture_images("images")
