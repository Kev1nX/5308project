import torch
from torchvision import models, transforms
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from network import ResNet18
import time
import os

images_folder_path = "images"
model_weights_path = "test.pth"

train_transform = transforms.Compose([
    transforms.Resize(500),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.564108, 0.50346, 0.427237), (0.20597, 0.206595, 0.21542))
])

def load_model(weights_path):
    model = ResNet18()
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    model.eval()  # Set the model to evaluation mode
    return model

def classify_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = train_transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = output.max(1)
    return predicted.item()

def get_first_image(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            return os.path.join(folder_path, file_name)
    return None

class ImageChangeHandler(FileSystemEventHandler):
    def __init__(self, model, folder_path):
        self.model = model
        self.folder_path = folder_path

    def on_created(self, event):
        if event.src_path.startswith(self.folder_path):
            image_path = get_first_image(self.folder_path)
            if image_path:
                print("New image detected. Classifying...")
                result = classify_image(self.model, image_path)
                print(f"Predicted class: {result}")
            
            filename: str = os.path.basename(image_path)
            filename = filename[:filename.rfind(".")]
            try:
                int(filename)
                os.remove(image_path)
            except:
                pass

model = load_model(model_weights_path)
event_handler = ImageChangeHandler(model, images_folder_path)
observer = Observer()
observer.schedule(event_handler, path=images_folder_path, recursive=False)

try:
    print("Monitoring for new images. Press Ctrl+C to stop.")
    observer.start()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
