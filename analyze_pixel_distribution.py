import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_pixel_distribution(folder_path):
    pixel_counts = np.zeros(256, dtype=int)
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
                    pixel_counts += hist
                    image_count += 1
    if image_count > 0:
        avg_pixel_counts = pixel_counts / image_count
        return avg_pixel_counts
    else:
        return None


methods = [
    {
        'folder_path': 'E:/0.dataset\MSTAR_jpg_x301\SOC/train/2S1',
        'color': 'purple',
        'label': 'Original'
    },
    {
        'folder_path': 'analyze_pixel_distribution_res/cave_gan/2s1',
        'color': 'yellow',
        'label': 'CVAE_GAN'
    },
    {
        'folder_path': 'analyze_pixel_distribution_res/acgan/2s1',
        'color': 'green',
        'label': 'ACGAN'
    },
    {
        'folder_path': 'analyze_pixel_distribution_res/ours/2s1',
        'color': 'blue',
        'label': 'Ours'
    }
]

results = []
for method in methods:
    folder_path = method['folder_path']
    label = method['label']
    result_file_path = f"analyze_pixel_distribution_res/{label}_result.json"

    if os.path.exists(result_file_path):
        with open(result_file_path, 'r') as file:
            data = json.load(file)
            result = np.array(data['result'])
            color = method['color']
            label = method['label']
            results.append((result, color, label))
    else:
        result = analyze_pixel_distribution(folder_path)
        if result is not None:
            color = method['color']
            results.append((result, color, label))
            result_data = {
                'result': result.tolist(),
                'color': color,
                'label': label
            }
            os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
            with open(result_file_path, 'w') as file:
                json.dump(result_data, file)  

