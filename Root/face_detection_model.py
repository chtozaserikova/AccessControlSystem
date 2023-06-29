import torch
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
# from google.colab.patches import cv2_imshow
import random
import PIL.ImageOps   
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from ultralytics import YOLO


# Параметры модели

# yolov5l_model.conf = 0.5  # порог уверенности модели для фильтрации предсказаний 
# yolov5l_model.iou = 0.4  # пороговое значение NMS IoU
# yolov5l_model.agnostic = False  # обнаружение объектов, не зависящих от класса
# yolov5l_model.multi_label = False  # несколько классов на один bbox
# yolov5l_model.classes = [0]  # выбор классов из COCO датасета, нулевой класс = persons
# yolov5l_model.max_det = 10  # максимальное количество обнаружений на одно изображение
# yolov5l_model.amp = True  # атоматический вывод смешанной точности


import torch
from PIL import Image
import torchvision.transforms as transforms


class FaceDetectionModel:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)
        self.model.eval()

    def detect_faces(self, image_path):
        image = Image.open(image_path).convert('RGB')

        # Преобразование изображения
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0).to(self.device)

        # Обнаружение объектов на изображении с использованием модели YOLO
        results = self.model(image)

        # Получение координат обнаруженных лиц
        faces_boxes = results.xyxy[0].cpu().numpy()

        # Обрезание и сохранение лиц на изображении
        cropped_faces = []
        for face_box in faces_boxes:
            x1, y1, x2, y2, _ = face_box
            cropped_face = image[:, :, int(y1):int(y2), int(x1):int(x2)]
            cropped_faces.append(cropped_face)

        return cropped_faces