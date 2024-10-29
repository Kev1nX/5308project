import time
import cv2
from datetime import datetime
import os
from threading import Event

def capture_images(folder_path, running:Event=None):
    # Open a connection to the default camera
    cap = cv2.VideoCapture(0)

    timer = time.time()
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    try:
        while running is None or running.is_set():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if ret:
                # Generate a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
                filename = f"{timestamp}.jpg"
                file_path = os.path.join(folder_path, filename)
                
                # Save the image to the specified folder
                cv2.imwrite(file_path, frame)

                if time.time() - timer > 1:
                    timer = time.time()
                    for file in os.listdir(folder_path):
                        if os.path.getmtime(file) - timer > 1:
                            os.remove(file)

            else:
                print("Error: Could not capture image.")
    except KeyboardInterrupt:
        pass
    
    # Release the camera
    cap.release()

if __name__ == "__main__":
    capture_images("images")