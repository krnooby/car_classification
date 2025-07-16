# ✅ ========== 사용된 모델 ==========
# ConvNeXt-Tiny

# 사전 학습 가중치: ConvNeXt_Tiny_Weights.IMAGENET1K_V1

# Classifier 구성:

# nn.Sequential(
#     nn.Flatten(),
#     nn.LayerNorm((768,), eps=1e-06),
#     nn.Dropout(p=0.3),
#     nn.Linear(in_features=768, out_features=NUM_CLASSES)
# )
# EMA 모델 사용: best_model_ema.pth 로드

# 디바이스 설정: cuda (가능 시), fallback은 cpu

# ✅ ========== 데이터 경로 및 구조 ==========

# ROOT_DIR: /home/project/car_classification/data

# TEST_CSV: test.csv

# sample_submission.csv: 클래스 순서 및 ID 참조용

# 최종 제출 경로: convnexttiny_ema_submission.csv

# ✅ ========== Test Time Augmentation (TTA) ==========
# 총 3가지 transform 사용:

# 기본 Resize (Baseline)

# transforms.Resize((224, 224))
# transforms.ToTensor()
# transforms.Normalize(...)
# Horizontal Flip

# transforms.Resize((224, 224))
# transforms.RandomHorizontalFlip(p=1.0)
# transforms.ToTensor()
# transforms.Normalize(...)
# Random Rotation (15도)

# transforms.Resize((224, 224))
# transforms.RandomRotation(15)
# transforms.ToTensor()
# transforms.Normalize(...)
# ✅ ========== Dataset 클래스 정의 ==========

# class TTADataset(Dataset):
#     def __init__(self, image_list, transform=None):
#         self.image_list = image_list  # (ID, 이미지 경로) 튜플
#         self.transform = transform

#     def __getitem__(self, idx):
#         image_id, path = self.image_list[idx]
#         image = Image.open(path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, image_id
# ✅ ========== 추론 전략 ==========

# TTA별로 추론 → softmax 확률 벡터 저장

# 총 3개 TTA 확률을 np.mean(stack, axis=0)으로 평균

# 평균된 확률로 최종 예측 결과 도출

# ✅ ========== 결과 저장 구조 ==========

# sample_submission에서 클래스 순서 불러옴

# ID + 각 클래스별 예측 확률

# 저장 파일: convnexttiny_ema_submission.csv

# 저장 옵션: encoding='utf-8-sig', float_format 지정 없음

# ✅ ========== 기타 세부 사항 ==========

# 배치 사이즈: 32

# DataLoader는 shuffle=False로 고정

# 이미지 경로: test_df['img_path'] 기준으로 ROOT_DIR 내부 경로 조합

# EMA 모델 사용하여 예측 정확도 향상 기대

# 최종 제출 점수 = 0.1989171572

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import Dataset, DataLoader

# 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = "/home/project/car_classification/data"
MODEL_PATH = "/home/project/best_model_ema.pth"  # ✅ EMA 모델 경로 사용
TEST_CSV = os.path.join(ROOT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(ROOT_DIR, "sample_submission.csv")
SUBMIT_PATH = os.path.join(ROOT_DIR, "convnexttiny_ema_submission.csv")

# 클래스 이름 추출
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
class_names = sample_submission.columns[1:].tolist()  # 'ID' 제외
NUM_CLASSES = len(class_names)

# 모델 정의 및 EMA 모델 로드
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=768, out_features=NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # ✅ EMA 모델 로드
model.to(DEVICE)
model.eval()

# TTA용 전처리 정의
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
]

# 단일 이미지 경로 기반 테스트셋
class TTADataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list  # [(ID, img_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id, path = self.image_list[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_id

# 테스트 데이터 준비
test_df = pd.read_csv(TEST_CSV)
image_list = [(row['ID'], os.path.join(ROOT_DIR, row['img_path'])) for _, row in test_df.iterrows()]

# TTA 추론
tta_outputs = []
for tta_tf in tta_transforms:
    dataset = TTADataset(image_list, transform=tta_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    probs_list = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Inference with {tta_tf.transforms[-1].__class__.__name__}"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probs_list.extend(probs)

    tta_outputs.append(np.array(probs_list))

# 평균 확률 계산
final_probs = np.mean(np.stack(tta_outputs), axis=0)

# 결과 저장
submission_df = pd.DataFrame(final_probs, columns=class_names)
submission_df.insert(0, 'ID', test_df['ID'])
submission_df = submission_df[sample_submission.columns]
submission_df.to_csv(SUBMIT_PATH, index=False, encoding='utf-8-sig')
print(f"✅ TTA 추론 완료. 결과 저장 위치: {SUBMIT_PATH}")
