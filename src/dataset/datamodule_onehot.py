from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import pandas as pd
import os
import hydra
import sys

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from omegaconf import DictConfig
from src.dataset.dataset_onehot import DocumentDataset_onehot
from src.dataset.dataset import DocumentDataset
from src.dataset.testdataset import TestDataset
from src.dataset.analyzedataset import AnalyzeDataset
from src.transform.custom_transform_ratio import get_augraphy_transform, get_grid_dropout, get_horizontal_flip, get_transform_brightness, get_transform_coarse_dropout, get_transform_img_comp, get_transform_rotation, get_transform_gaussNoise, get_transform_blur, get_transform_shadow, get_transform_norm_tensor, get_vertical_flip
# from src.transform.custom_transform import get_augraphy_transform, get_transform_rotation, get_transform_gaussNoise, get_transform_blur, get_transform_shadow, get_test_transform
# from src.transform.custom_transform import get_transform_custom
from src.utils.augmentation import custom_collate_fn

class DocumentDataModule(pl.LightningDataModule):
    def __init__(self, 
    data_dir=f"{ROOT_DIR}/data", 
    batch_size=32, 
    num_workers=4, 
    persistent_workers=True,
    pin_memory=True,
    val_split=0.2, 
    image_size=(456, 456), 
    image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, 
    apply_transform_prob=0.8,
    mixup_alpha = 0.4,
    aug_ratio = [0, 0.2, 0.8],
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.image_size = image_size
        self.image_normalization = image_normalization
        self.apply_transform_prob = apply_transform_prob
        self.mixup_alpha = mixup_alpha
        self.aug_ratio = aug_ratio
        # self.df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.df_aug = pd.read_csv(os.path.join(self.data_dir, "train_aug.csv"))

        # AugraphyPipeline 정의
        self.aug_pipeline = get_augraphy_transform()


        self.transform_test = [get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        # 회전 변환
        self.transform_rotation = [get_transform_rotation(p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        # 밝기 + 회전
        self.transform_brightness = [get_transform_brightness(brightness_limit=(0,0.25), p=self.apply_transform_prob), get_transform_rotation(p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        # 블러 + 밝기 + 회전
        self.transform_blur = [get_transform_blur(blur_limit=(2,3), p=self.apply_transform_prob), get_transform_brightness(brightness_limit=(0,0.25), p=self.apply_transform_prob), get_transform_rotation(p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        # 그림자 + 회전
        self.transform_shadow = [get_transform_shadow(p=self.apply_transform_prob), get_transform_rotation(p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        #가우스 노이즈 + 회전 + 밝기
        self.transform_gaussNoise = [get_transform_gaussNoise(std_range=(0.1, 0.2), p=self.apply_transform_prob), get_transform_rotation(p=self.apply_transform_prob), get_transform_brightness(brightness_limit=(0,0.25), p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        #회전 + JPEG 압축
        self.transform_img_comp = [get_transform_rotation(p=0.5), get_transform_img_comp(compression_type='jpeg', quality_range=(20, 40), p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        #coarse dropout + 회전
        self.transform_coarse_dropout = [get_transform_coarse_dropout(num_holes_range=[2, 4], hole_height_range=[0.1, 0.2], hole_width_range=[0.1, 0.12], fill=0, p=self.apply_transform_prob), get_transform_rotation(p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        #grid dropout + 밝기 + 회전
        self.transform_grid_dropout = [get_grid_dropout(ratio=0.5, random_offset=True, shift_xy=[0, 0], fill=0, p=self.apply_transform_prob), get_transform_rotation(p=self.apply_transform_prob), get_transform_brightness(brightness_limit=(0,0.25), p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]

        #grid dropout + horizontal flip + 밝기 
        self.transform_grid_dropout_horizontal_flip = [get_grid_dropout(ratio=0.5, random_offset=True, shift_xy=[0, 0], fill=0, p=self.apply_transform_prob), get_horizontal_flip(p=1.0), get_transform_rotation(p=self.apply_transform_prob), get_transform_brightness(brightness_limit=(0,0.25), p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]
        
        #grid dropout + vertical flip + 밝기
        self.transform_grid_dropout_vertical_flip = [get_grid_dropout(ratio=0.5, random_offset=True, shift_xy=[0, 0], fill=0, p=self.apply_transform_prob), get_vertical_flip(p=1.0), get_transform_rotation(p=self.apply_transform_prob), get_transform_brightness(brightness_limit=(0,0.25), p=self.apply_transform_prob), get_transform_norm_tensor(image_size=self.image_size, image_normalization=self.image_normalization)]
        """
        self.transform_custom = [get_transform_custom(image_size=self.image_size, image_normalization=self.image_normalization)]
        """

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.analyze_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # train_df, val_df = train_test_split(
            #     self.df_aug,
            #     test_size=self.val_split,
            #     stratify=self.df_aug["target"],
            #     random_state=42
            # )
            train_df = self.df_aug[self.df_aug["train"] == 1]
            val_df = self.df_aug[self.df_aug["train"] == 0]
            
            train_dataset_rotation = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_rotation)
            train_dataset_gaussNoise = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_gaussNoise)
            train_dataset_blur = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_blur)
            train_dataset_brightness = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_brightness)
            # train_dataset_img_comp = DocumentDataset(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_img_comp)
            train_dataset_coarse_dropout = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_coarse_dropout)
            train_dataset_grid_dropout = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_grid_dropout)
            train_dataset_grid_dropout_horizontal_flip = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_grid_dropout_horizontal_flip)
            train_dataset_grid_dropout_vertical_flip = DocumentDataset_onehot(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_grid_dropout_vertical_flip)
            # self.train_dataset_custom = DocumentDataset(train_df, self.data_dir, aug_pipeline=None, transform=self.transform_custom)
            self.train_dataset = torch.utils.data.ConcatDataset([train_dataset_rotation, train_dataset_gaussNoise, train_dataset_blur, train_dataset_brightness, train_dataset_coarse_dropout, train_dataset_grid_dropout, train_dataset_grid_dropout_horizontal_flip, train_dataset_grid_dropout_vertical_flip])
            
            self.val_dataset_no_augraphy = DocumentDataset(val_df, self.data_dir, aug_pipeline=None, transform=self.transform_rotation)
            self.val_dataset_augraphy = DocumentDataset(val_df, self.data_dir, aug_pipeline=self.aug_pipeline, transform=self.transform_rotation)
            self.val_dataset = torch.utils.data.ConcatDataset([self.val_dataset_no_augraphy, self.val_dataset_augraphy])

        if stage == "predict" or stage is None:
            self.test_dataset = TestDataset(self.data_dir, transform=self.transform_test)

        if stage == "analyze":
            # train_df, val_df = train_test_split(
            #     self.df_aug,
            #     test_size=self.val_split,
            #     stratify=self.df_aug["target"],
            #     random_state=42
            # )
            # train_df = self.df_aug[self.df_aug["train"] == 1]
            val_df = self.df_aug[self.df_aug["train"] == 0]
            
            self.analyze_dataset_no_augraphy = AnalyzeDataset(val_df, self.data_dir, aug_pipeline=None, transform=self.transform_rotation)
            self.analyze_dataset_augraphy = AnalyzeDataset(val_df, self.data_dir, aug_pipeline=self.aug_pipeline, transform=self.transform_rotation)
            self.analyze_dataset = torch.utils.data.ConcatDataset([self.analyze_dataset_no_augraphy, self.analyze_dataset_augraphy])
        # full_dataset = ImageFolder(os.path.join(self.data_dir, "train"))
        # train_label_csv = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        # targets = full_dataset.targets  # 클래스 인덱스 리스트
        # indices = np.arange(len(full_dataset))

        # train_idx, val_idx = train_test_split(
        #     indices,
        #     test_size=self.val_split,
        #     stratify=targets,
        #     random_state=42
        # )

        # self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        # self.val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

        # # transform은 Subset 내부의 dataset에 직접 설정
        # full_dataset.transform = self.transform

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not initialized")
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers, pin_memory=self.pin_memory)
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=self.num_workers,
          persistent_workers=self.persistent_workers,
          pin_memory=self.pin_memory,
          collate_fn=lambda batch: custom_collate_fn(batch, alpha=self.mixup_alpha, aug_ratio=self.aug_ratio)
        )
    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("val_dataset is not initialized")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers, pin_memory=self.pin_memory)
    
    def analyze_dataloader(self):
        if self.analyze_dataset is None:
            raise ValueError("analyze_dataset is not initialized")
        return DataLoader(self.analyze_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers, pin_memory=self.pin_memory)
    
    def predict_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("test_dataset is not initialized")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dm = DocumentDataModule(**cfg.data)
    dm.setup()

if __name__ == "__main__":
    main()