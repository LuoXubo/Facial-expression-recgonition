"""
@Description :   Emotion recognition with face detection of OpenCV.
                 Singel and multiple faces are supported.
@Author      :   Xubo Luo 
@Time        :   2024/04/23 14:54:23
"""

import matplotlib.pyplot as plt
import torch
import warnings
import tqdm
import time
import os
import argparse
import cv2
import numpy as np
from PIL import Image
from utils.DataLoader import get_dataloader
from models.resnet.resnet_50 import ResNet50
from models.resnet.resnet_101 import ResNet101
from models.resnet.resnet_152 import ResNet152
from models.efficientnet.model import EfficientNet
from models.vit.model import ViT
warnings.filterwarnings("ignore")

# configs
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='resnet50', help='model name (resnet50, resnet101 or resnet152)')
    parser.add_argument("--weight_path", type=str, default=None, help='pretrained model path')
    parser.add_argument("--image_path", type=str, default='testimg.jpeg', help='image path')
    args = parser.parse_args()

    method = args.method
    weight_path = args.weight_path
    image_path = args.image_path


    # load data
    print('Loading data...')
    image = Image.open(image_path)
    image = image.convert('L')

    # load model
    print('------------------------------------')
    if method == 'resnet50':
        model = ResNet50
    elif method == 'resnet101':
        model = ResNet101
    elif method == 'resnet152':
        model = ResNet152
    elif method == 'efficientnet0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=7)
    elif method == 'efficientnet4':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=7)
    elif method == 'efficientnet7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=7)
    elif method == 'vit':
        model = ViT(
        image_size = 100,
        patch_size = 10,
        num_classes = 7,
        dim = 1024,
        depth = 3,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    else:
        raise ValueError('Invalid model name!')
    
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print('%s loaded successfully!' % method)

    # face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    # emotion recognition
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w, :]
        single_face = Image.fromarray(roi_color)
        single_face = single_face.resize((100, 100))
        single_face = np.array(single_face)
        single_face = single_face.reshape(1, 3, 100, 100)
        single_face = single_face / 255.0
        single_face = torch.tensor(single_face, dtype=torch.float32)
        emotion = model(single_face)
        emotion = torch.argmax(emotion, dim=1)
        emotion = emotion.item()
        cv2.putText(img, emotions[emotion], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    