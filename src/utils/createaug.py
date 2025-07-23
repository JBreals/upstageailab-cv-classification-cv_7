import os
import cv2
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import time
from augraphy import AugraphyPipeline

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.transform.augraphy_transform import delaunay, folding, ink_bleed, lines_degradation, noise_texturize, pattern

docs = {
  'docs': [1,3,4,6,7,10,11,12,13,14,15]

}

def create_aug_target(df, target_class, aug_pipeline, number = 20, aug_name = 'aug', split = True):
  current_time = time.strftime("%Y%m%d_%H%M%S")

  if split:
    df = df[df["train"] == 1]

  df = df[df["target"].isin(target_class)]

   # 클래스별로 number만큼 샘플링
  sampled_df = (
      df.groupby("target", group_keys=False)
        .apply(lambda x: x.sample(n=min(number, len(x))))
  )
  
  img_names, targets = sampled_df["ID"], sampled_df["target"]

  records = []
  for img_name, target in zip(img_names, targets):
    img_path = os.path.join(ROOT_DIR, "data", "train", img_name)
    img = cv2.imread(img_path)

    if img is None:
            print(f"[경고] 이미지 로드 실패: {img_path}")
            continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = img_name.replace(".jpg", "")
    aug_img = aug_pipeline(img)
    file_name = f"{img_name}_{current_time}_{aug_name}.jpg"

    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(ROOT_DIR, "data", "train", file_name), aug_img)
    records.append({"ID": file_name, 'aug':1, "target": target, 'train': 1, 'aug_name': aug_name })

  record_df = pd.DataFrame(records)

  aug_csv_path = os.path.join(ROOT_DIR, "data", "train_aug.csv")

  if os.path.exists(aug_csv_path):
    existing_df = pd.read_csv(aug_csv_path)
  else:
    existing_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_split.csv"))
    existing_df['aug'] = 0
    existing_df['aug_name'] = '-'
    
  combined_df = pd.concat([existing_df, record_df], ignore_index=True)
  combined_df.to_csv(aug_csv_path, index=False)

  print(f"[완료] 총 {len(record_df)}개의 증강 이미지가 저장되었고, train_aug.csv가 갱신되었습니다.")


def create_aug(df, aug_pipeline, number = 20, aug_name = 'aug'):
  target_class = df["target"].unique()
  create_aug_target(df, target_class, aug_pipeline, number, aug_name)


if __name__ == "__main__":
  df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_split.csv"))

  ink_noise_pipeline = AugraphyPipeline(
    ink_phase=[ink_bleed(p=1.0), noise_texturize(p=1.0)],
    paper_phase=[],
    post_phase=[]
  )

  folding_pipeline = AugraphyPipeline(
    ink_phase=[],
    paper_phase=[],
    post_phase=[folding(p=1.0)]
  )

  delaunay_pipeline = AugraphyPipeline(
    ink_phase=[],
    paper_phase=[],
    post_phase=[delaunay(p=1.0)]
  )

  pattern_pipeline = AugraphyPipeline(
    ink_phase=[],
    paper_phase=[],
    post_phase=[pattern(p=1.0)]
  )
  # create_aug_target(df, docs['docs'], ink_noise_pipeline, 40, "ink_noise")
  create_aug(df, delaunay_pipeline, 40, "delaunay")