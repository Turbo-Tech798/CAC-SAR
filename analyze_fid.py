import torch
import itertools
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
import os
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def collect_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

inception_model = torchvision.models.inception_v3(pretrained=True).to("cuda:0")
model_fid_values = []

classes = ('2s1', 'bmp2', 'brdm2', 'btr60', 'btr70', 'd7', 't62', 't72', 'zil131', 'zsu23-4')
path_ours = 'analyze_fid/ours'
path_acgan = 'analyze_fid/acgan'
path_cvaegan = 'analyze_fid/cvae_gan'
real_images_folder = f"enhance_datasets_asc_128_imban/train"
results_ours = []
results_acgan = []
results_cvaegan = []

for class_name in classes:
    raw_class_path = os.path.join(real_images_folder, class_name)

    ours_class_path = os.path.join(path_ours, class_name)
    fid_value = fid_score.calculate_fid_given_paths([raw_class_path, ours_class_path], batch_size=512, device='cuda:0', dims=2048, num_workers=0)
    results_ours.append(fid_value)

    acgan_class_path = os.path.join(path_acgan, class_name)
    fid_value = fid_score.calculate_fid_given_paths([raw_class_path, acgan_class_path], batch_size=512, device='cuda:0',
                                                    dims=2048, num_workers=0)
    results_acgan.append(fid_value)

    cvaegan_class_path = os.path.join(path_cvaegan, class_name)
    fid_value = fid_score.calculate_fid_given_paths([raw_class_path, cvaegan_class_path], batch_size=512, device='cuda:0',
                                                    dims=2048, num_workers=0)
    results_cvaegan.append(fid_value)