import pandas as pd

df = pd.read_csv("runs/detect/Indoor_Model/results.csv")

best = df.loc[df["metrics/mAP50(B)"].idxmax()]

print(best[[
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]])
