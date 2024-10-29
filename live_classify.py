import threading
import cv2
import torch
from torch import Tensor
from torchvision import models, transforms
from PIL import Image
from classify import SquarePad
from network import ResNet18
import time
from arducar_com import update_car

images_folder_path = "images"
model_weights_path = "99.04.pth"

# 99.04
train_transform = transforms.Compose([
    transforms.Resize(500),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.564108, 0.50346, 0.427237), (0.20597, 0.206595, 0.21542))
])

# 97.27
# train_transform = transforms.Compose([
#     SquarePad(),
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.564108, 0.50346, 0.427237), (0.20597, 0.206595, 0.21542))
# ])

def load_model(weights_path):
    model = ResNet18()
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    model.eval()  # Set the model to evaluation mode
    return model

def classify_camera(cap, model):
    ret, frame = cap.read()
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img_pil.convert("RGB")  # Convert OpenCV image to PIL format
    img = train_transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output: Tensor = model(img)
        _, predicted = output.max(1)
        # print(output[0][predicted.item()], predicted.item())
        if output[0][predicted.item()] < 7:
            return None
    return predicted.item()

model = load_model(model_weights_path)
cap = cv2.VideoCapture(0)

while True:
    result = classify_camera(cap, model)
    if result is not None:
        update_car(result)
        print(f"Predicted class: {result}")

