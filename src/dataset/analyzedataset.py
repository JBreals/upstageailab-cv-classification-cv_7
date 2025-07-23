from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np

class AnalyzeDataset(Dataset):
    def __init__(self, df, data_dir, aug_pipeline=None, transform=None):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        self.aug_pipeline = aug_pipeline
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name, label = row["ID"], row["target"]
        img_path = os.path.join(self.data_dir, "train", img_name)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.aug_pipeline:
            image = self.aug_pipeline(image)

        if self.transform:
            for transform in self.transform:
                image = transform(image=image)
                image = image['image']

        return image, img_name, label  
