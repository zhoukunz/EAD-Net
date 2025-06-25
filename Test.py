import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import cv2
import sys
import json
import time
import torch
import torch.optim as optim
# from ViTmodel import VisionTransformer
# from Swinmodel import SwinTransformer
# from resnet import *
from PIL import Image
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import accuracy_score
from scipy.ndimage import zoom

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.samples = []
        for label in ['0', '1', '2']:  # Update for 3 classes
            class_dir = os.path.join(self.directory, label)
            for filename in os.listdir(class_dir):
                if filename.endswith('.nii.gz'):
                    self.samples.append((os.path.join(class_dir, filename), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        nii_image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(nii_image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long),path


def evaluate_model(model, test_loader, device):
    model.eval()  
    pre = []
    test_loss = 0.0
    correct = 0
    all_predictions = []
    all_labels = []
    all_paths = []

    with torch.no_grad():  
        pre = []  
        prob = []  

        for data, label, paths in test_loader:
            data = data.to(device)
            label = label.data.cpu().numpy()[0]
            output = model(data)

            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_label = torch.max(output, dim=1)[1].cpu().numpy()[0]
            pre.append([paths[0].split('/')[-1].split('.')[0], label, pred_label])
            prob.append(probabilities)
        df = pd.DataFrame(data=pre, columns=['name', 'label_true', 'pred_label'])
        prob_df = pd.DataFrame(prob, columns=[f'class_{i}' for i in range(probabilities.shape[0])])
        result_df = pd.concat([df, prob_df], axis=1)
        result_df.to_csv('/home/fuying/DATASET/HS/test_ConvNet.csv', index=False)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    val_path = './image/'
    test_dataset = GetLoader(val_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    len_test = len(test_dataset)

    # model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=3)
    # model = SwinTransformer(in_chans=14,
    #                       patch_size=4,
    #                       window_size=7,
    #                       embed_dim=192,
    #                       depths=(2, 2, 18, 2),
    #                       num_heads=(6, 12, 24, 48),
    #                       num_classes=3,
    #                       )
    # model = VisionTransformer(img_size=224,
    #                         patch_size=14,
    #                         embed_dim=1280,
    #                         depth=32,
    #                         num_heads=16,
    #                         representation_size=1280,
    #                         num_classes=3)
    model.load_state_dict(torch.load('./logs/best.pth'))  
    model.to(device)

    evaluate_model(model, test_loader,device)
