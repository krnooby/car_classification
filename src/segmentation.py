import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamPredictor

# ========= 설정 =========
INPUT_DIR = "/home/project/car_classification/data/train_plate_fixed"
OUTPUT_DIR = "/home/project/car_classification/data/train_segmented"
CHECKPOINT_PATH = "/home/project/car_classification/sam_vit_b.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = "vit_b"
MIN_MASK_AREA_RATIO = 0.01  # 최소 마스크 비율 (1%)

# ========= 대비 향상 함수 =========
def enhance_contrast(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

# ========= 모델 로딩 =========
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# ========= 마스크 적용 함수 =========
def segment_and_save(img_path, save_path):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"[오류] 이미지 로딩 실패: {img_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = enhance_contrast(image_rgb)  # 대비 향상
    predictor.set_image(image_rgb)

    h, w = image_rgb.shape[:2]

    input_points = np.array([
        [w // 2, h // 2],                  # 중앙
        [w // 3, h // 2],                  # 왼쪽
        [2 * w // 3, h // 2],              # 오른쪽
        [w // 2, 3 * h // 4],              # 하단
        [w // 2, h // 4],                  # 상단
        [w // 4, h // 4],                  # 좌상단
        [3 * w // 4, 3 * h // 4],          # 우하단
        [w // 2, h - 10],                  # 하단 경계 근처
    ])
    input_labels = np.ones(len(input_points), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    # 가장 큰 마스크 선택
    best_mask = None
    max_area = 0
    for m in masks:
        area = m.sum()
        if area > max_area:
            best_mask = m
            max_area = area

    if max_area < MIN_MASK_AREA_RATIO * (h * w):
        print(f"[경고] 마스크 너무 작음 → 원본 저장: {img_path}")
        segmented = image_rgb.copy()
    else:
        segmented = image_rgb.copy()
        segmented[~best_mask] = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

# ========= 전체 이미지 처리 =========
for fname in tqdm(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    input_img = os.path.join(INPUT_DIR, fname)
    output_img = os.path.join(OUTPUT_DIR, fname)
    segment_and_save(input_img, output_img)

print("✅ 세그멘테이션 완료! 저장 위치:", OUTPUT_DIR)
