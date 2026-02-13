import torch
import itertools
import torchvision
import torchvision.transforms as transforms
from torchmetrics.image.inception import InceptionScore
import os
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from PIL import Image

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

def load_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)
    images = torch.stack(images)
    images = (images * 255).clamp(0, 255).to(torch.uint8)
    return images

classes = ('2s1', 'bmp2', 'brdm2', 'btr60', 'btr70', 'd7', 't62', 't72', 'zil131', 'zsu23-4')
path_ours = 'analyze_fid/ours'
path_acgan = 'analyze_fid/acgan'
path_cvaegan = 'analyze_fid/cvae_gan'

results_ours = []
results_acgan = []
results_cvaegan = []

inception_score = InceptionScore(normalize=False).to("cuda:1")

for class_name in classes:
    ours_class_path = os.path.join(path_ours, class_name)
    ours_image_paths = collect_image_paths(ours_class_path)
    ours_images = load_images(ours_image_paths).to("cuda:1")
    inception_score.update(ours_images)
    mean, std = inception_score.compute()
    results_ours.append(mean.item())
    inception_score.reset()
    acgan_class_path = os.path.join(path_acgan, class_name)
    acgan_image_paths = collect_image_paths(acgan_class_path)
    acgan_images = load_images(acgan_image_paths).to("cuda:1")
    inception_score.update(acgan_images)
    mean, std = inception_score.compute()
    results_acgan.append(mean.item())
    inception_score.reset()
    cvaegan_class_path = os.path.join(path_cvaegan, class_name)
    cvaegan_image_paths = collect_image_paths(cvaegan_class_path)
    cvaegan_images = load_images(cvaegan_image_paths).to("cuda:1")
    inception_score.update(cvaegan_images)
    mean, std = inception_score.compute()
    results_cvaegan.append(mean.item())
    inception_score.reset()

bar_width = 0.2
r1 = np.arange(len(classes))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

print(results_ours)
print(results_acgan)
print(results_cvaegan)