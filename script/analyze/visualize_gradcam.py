from hydra.utils import instantiate
import pandas as pd
import os
import sys
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
from pytorch_grad_cam.base_cam import BaseCAM

from collections import defaultdict
from tqdm import tqdm

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch.utils.data import DataLoader

def visualize_gradcam(model, image_tensors_dict, title='correct', device="cuda", target_layer=None):

    num_classes = len(image_tensors_dict)
    valid_classes = [label for label, tensors in image_tensors_dict.items() if len(tensors) > 0]
    n_rows = len(valid_classes)
    n_cols = 5  # 최대 5개

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = [axes]  # 1행일 경우 axes를 리스트로 변환

    for row_idx, label in enumerate(valid_classes):
        image_tensors = image_tensors_dict[label][:5]  # 최대 5개
        targets = [ClassifierOutputTarget(label)]

        for col_idx, image_tensor in enumerate(image_tensors):
            # Grad-CAM 준비
            cam = GradCAM(model=model, target_layers=[target_layer])
            input_tensor = image_tensor.unsqueeze(0).to(device)

            # Normalize for visualization
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            ax = axes[row_idx][col_idx] if n_rows > 1 else axes[0][col_idx]
            ax.imshow(visualization)
            ax.axis('off')
            ax.set_title(f'{label} {title} #{col_idx+1}')

        # 나머지 빈 칸 비우기
        for col_idx in range(len(image_tensors), n_cols):
            ax = axes[row_idx][col_idx] if n_rows > 1 else axes[0][col_idx]
            ax.axis('off')

    plt.tight_layout()
    plt.show()
  