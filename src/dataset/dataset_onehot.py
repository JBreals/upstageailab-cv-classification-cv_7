from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np
import torch.nn.functional as F
import torch

class DocumentDataset_onehot(Dataset):
    def __init__(self, df_subset, data_dir, aug_pipeline=None, transform=None):
        self.df = df_subset
        self.data_dir = data_dir
        self.transform = transform
        self.aug_pipeline = aug_pipeline
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name, label = row["ID"], row["target"]
        one_hot = F.one_hot(torch.tensor(label), num_classes=17).float()
        img_path = os.path.join(self.data_dir, "train", img_name)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.aug_pipeline:
            image = self.aug_pipeline(image)

        # # 확률적 augraphy 증강 적용
        # prob = random.random()
        # if self.aug_pipeline and (prob >= self.apply_transform_prob):
        #     image = self.aug_pipeline(image)

        if self.transform:
            for transform in self.transform:
                image = transform(image=image)
                image = image['image'] 
        

        return image, one_hot
