from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def result_analysis(df):
  report = classification_report(df['label'], df['pred'], output_dict=True)
  report_df = pd.DataFrame(report).transpose().reset_index()
  report_df = report_df.rename(columns={"index": "class"})

  # 정수 클래스만 필터링 (macro avg, weighted avg 제외)
  report_df = report_df[report_df['class'].str.isdigit()]
  report_df['class'] = report_df['class'].astype(int)

  report_df = report_df.sort_values(by='class').reset_index(drop=True)

  display(report_df)


def confusion_matrix_analysis(df):
  labels = sorted(set(df['label'].unique()) | set(df['pred'].unique()))
  conf_mat = confusion_matrix(df['label'], df['pred'], labels=labels)

  plt.figure(figsize=(12, 10))
  sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
              xticklabels=labels, yticklabels=labels)
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix")
  plt.tight_layout()