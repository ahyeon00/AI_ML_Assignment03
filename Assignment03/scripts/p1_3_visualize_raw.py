
import os
import sys
import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.viser_test import PoseViser

# 원본 데이터 경로
DATA_PATH = os.path.join(
    ROOT, "data", "raw_keypoints", "HAERIN_HowSweet.npz"
)

VIDEO_PATH = os.path.join(
    ROOT, "data", "raw_videos", "HAERIN_HowSweet.mp4"
)

# 원본 3D 관절 데이터 불러오기
data = np.load(DATA_PATH, allow_pickle=True)["data"]

df = pd.DataFrame(
    data,
    columns=["frame", "landmark", "x", "y", "z", "visibility"]
)

# 프레임별 33개 관절 좌표 정리
pts_seq = []

for frame, group in df.groupby("frame"):
    group = group.sort_values("landmark")
    pts = group[["x", "y", "z"]].values
    pts_seq.append(pts)

raw_seq = np.array(pts_seq, dtype=np.float32)

# 원본 영상 FPS 확인
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if not np.isfinite(fps) or fps <= 0:
    fps = 30

# 데이터 정보 출력
print("Data shape:", raw_seq.shape)
print("Total detected frames:", len(raw_seq))
print("Landmarks:", raw_seq.shape[1])
print("Video FPS:", fps)

# 원본 모션 시각화
vis = PoseViser(fps=fps)
vis.play_sequence(raw_seq)
