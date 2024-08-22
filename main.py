#!/usr/bin/env python

import ultralytics
import torch
import torchvision

from ultralytics import YOLO


def run_model():
        
    # Create a new YOLO model from scratch
    model = YOLO("yolov8n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data="coco8.yaml", epochs=3)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model("https://ultralytics.com/images/bus.jpg")

if __name__ == "__main__":
    # not tested, dummy code from documentation
    run_model()