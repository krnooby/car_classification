import sys
sys.path.append("/home/project/yolov5")

import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression)

# ====== 설정 ======
IMG_DIR = "/home/project/car_classification/data/train"
SAVE_DIR = "/home/project/car_classification/data/train_plate_fixed"
WEIGHTS = "/home/project/yolov5/runs/train/plate_yolo/weights/best.pt"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 모델 로딩 ======
model = DetectMultiBackend(WEIGHTS, device=DEVICE)
model.eval()

# ====== 전체 클래스 폴더 순회 ======
image_extensions = [".jpg", ".jpeg", ".png"]

for class_name in tqdm(os.listdir(IMG_DIR), desc="클래스별 처리"):
    class_path = os.path.join(IMG_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    save_class_path = os.path.join(SAVE_DIR, class_name)
    os.makedirs(save_class_path, exist_ok=True)

    for fname in os.listdir(class_path):
        if not any(fname.lower().endswith(ext) for ext in image_extensions):
            continue

        img_path = os.path.join(class_path, fname)
        save_path = os.path.join(save_class_path, fname)

        # 이미지 로딩
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[오류] 이미지 로드 실패: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(img_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]

        # bbox 좌표 복원 및 마스킹
        if pred is not None and len(pred):
            # YOLOv5 최신 버전에선 pred는 원래 이미지 기준으로 나옴
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), -1)  # 흰색 박스

        cv2.imwrite(save_path, img_bgr)

print("✅ 번호판 제거 완료 → 저장 위치:", SAVE_DIR)
