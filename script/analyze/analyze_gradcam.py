from hydra.utils import instantiate
import pandas as pd
import os
import sys
import torch

from collections import defaultdict
from tqdm import tqdm

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch.utils.data import DataLoader

def analyze_gradcam(model, dm, fold=0):
  pred_list = defaultdict(lambda: {"img_name": [], "img": [], "pred": [], "label": []})

  dm.setup("analyze")
  analyze_dataset = dm.analyze_dataset
  if analyze_dataset is None:
    raise ValueError("analyze_dataset is not initialized")
  
  # 데이터로더 준비
  analyze_loader = DataLoader(analyze_dataset, batch_size=32, shuffle=False)
  with torch.no_grad():
    for batch in tqdm(analyze_loader):
      output = model.predict_step_analyze(batch)
      pred_list[f"{fold}"]["img_name"].extend(output["img_name"])
      pred_list[f"{fold}"]["pred"].extend(output["pred"])
      pred_list[f"{fold}"]["label"].extend(output["label"])
      pred_list[f"{fold}"]["img"].extend(output["img"])
  recoder = []
  img_storage = []
  
  for fold, values in pred_list.items():
    img_name = values["img_name"]
    pred = values["pred"]
    label = values["label"]
    img = values["img"]
    for img_name, pred, label, img in zip(img_name, pred, label, img):
      recoder.append({"fold": fold, "img_name": img_name, "pred": pred, "label": label})
      img_storage.append(img)

  df = pd.DataFrame(recoder).reset_index().rename(columns={"index": "id"})
  df_correct = df[df["pred"] == df["label"]]
  df_incorrect = df[df["pred"] != df["label"]]

  # 라벨별 대표 이미지 인덱스 (정답 그룹)
  correct_dict = {}
  for label, group in df_correct.groupby("label"):
      ids = group["id"].tolist()[:5]  # 최대 5개까지 가져오기
      correct_dict[label] = [img_storage[idx] for idx in ids]

  # 라벨별 대표 이미지 인덱스 (오답 그룹)
  incorrect_dict = {}
  for label, group in df_incorrect.groupby("label"):
      ids = group["id"].tolist()[:5]  # 최대 5개까지 가져오기
      incorrect_dict[label] = [img_storage[idx] for idx in ids]

  
  
  return df, correct_dict, incorrect_dict
  