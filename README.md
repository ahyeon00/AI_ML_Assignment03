# AI_ML_Assignment03
3D Motion Data Post Processing

## 1. Project Overview

MediaPipe BlazePose를 활용하여 영상에서 3D 모션 데이터를 추출하고, 높이 보정 및 추가 스무딩을 통해 모션 데이터의 시간적 안정성을 개선

## 2. Dataset

- Source: NewJeans HAERIN – How Sweet Dance
- Video: https://youtu.be/HjEV8G7RFMA
- Detected Frames: 578
- Landmarks: 33
- Video FPS: 30
- Data Shape: (578, 33, 3)

## 3. Post-Processing Methods

- Height Correction
- Savitzky–Golay Filter
- Window Length: 7
- Polynomial Order: 2
- Ground Re-alignment

## 4. Experimental Results

| Method | Motion Roughness (m) |
|---|---:|
| Original | 0.03904 |
| Height Correction | 0.04094 |
| Height Correction + Smoothing | 0.01802 |

추가 스무딩 적용 후 높이 보정 데이터 대비 Motion Roughness가 약 56.0% 감소하였다.

## 5. Visualization

- Viser 기반 3D 스켈레톤 시각화
- Matplotlib 기반 후처리 전후 동시 비교
- 프레임 선택, 재생 및 이미지 저장 기능
- Unity 기반 3D 캐릭터 애니메이션

## 6. Execution Order

1. `scripts/p1_create_raw_keypoints.py`
2. `scripts/p1_3_visualize_raw.py`
3. `scripts/p1_2_post_processing.py`
4. `scripts/p1_4_smoothing.py`
5. `scripts/p1_5_compare_motion.py`
6. `scripts/p1_6_analysis.py`
