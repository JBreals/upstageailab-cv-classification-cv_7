import os
import cv2
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import time
from augraphy import AugraphyPipeline, default_augraphy_pipeline

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def delete_aug_with_name(aug_name):
    csv_path = os.path.join(ROOT_DIR, "data", "train_aug.csv")
    image_dir = os.path.join(ROOT_DIR, "data", "train")

    # CSV 불러오기
    df = pd.read_csv(csv_path)
    df_aug = df[df["aug"] == 1]

    # 삭제 대상: 증강 이미지 중 ID에 aug_name이 포함된 것
    delete_mask = (df["ID"].str.contains(aug_name))
    delete_df_aug = df_aug[delete_mask]

    # 실제 이미지 삭제
    for img_name in delete_df_aug["ID"]:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"[삭제] {img_path}")
        else:
            print(f"[경고] 파일 없음: {img_path}")

    # 삭제 대상 제외하고 CSV 저장
    df = df[~delete_mask]
    df.to_csv(csv_path, index=False)

    print(f"[완료] {len(delete_df_aug)}개의 증강 이미지와 관련 정보가 삭제되었습니다.")

def delete_all_aug():
  csv_path = os.path.join(ROOT_DIR, "data", "train_aug.csv")
  df = pd.read_csv(csv_path)
  df = df[df["aug"] == 1]
  for img_name in df["ID"]:
    img_path = os.path.join(ROOT_DIR, "data", "train", img_name)
    if os.path.exists(img_path):
      os.remove(img_path)

  #csv 파일 삭제
  os.remove(csv_path)


if __name__ == "__main__":
  delete_all_aug()