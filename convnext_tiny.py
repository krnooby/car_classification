# ✅ ========== 사용된 모델 ==========
# ConvNeXt-Tiny

# 사전 학습 가중치: ConvNeXt_Tiny_Weights.IMAGENET1K_V1

# Classifier 구성:

# nn.Sequential(
#     nn.Flatten(),
#     nn.LayerNorm((768,), eps=1e-06),
#     nn.Dropout(p=0.3),
#     nn.Linear(768, NUM_CLASSES)
# )
# 출력 클래스 수: 396

# 디바이스 설정: cuda (가능 시), fallback은 cpu

# ✅ ========== 데이터 증강 (train) ==========

# transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
# transforms.RandomHorizontalFlip()
# transforms.RandomRotation(15)
# transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
# transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
# transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
# transforms.ToTensor()
# transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225])
# ✅ ========== 데이터 전처리 (val) ==========

# train 데이터를 8:2 비율로 나누어 validation set 구성

# val도 동일 transform 사용 (다르게 하고 싶으면 별도 정의 필요)

# ✅ ========== 학습 전략 ==========

# MixUp (Beta(0.4, 0.4))

# 입력 이미지와 라벨을 lam 비율로 섞어서 soft label 구성

# 두 번 forward → CE loss 각각 계산

# Dual forward + KL Loss

# 두 번 forward → log_softmax 비교 후 KLDivLoss로 유사도 측정

# Total Loss = 평균(CrossEntropy) + 0.5 * 평균(KLDiv)

# ✅ ========== 최적화 및 정규화 ==========

# Optimizer: SAM (Sharpness-Aware Minimization) + AdamW

# learning rate = 1e-4

# weight decay = 1e-4

# SAM ρ = 0.05

# Scheduler: CosineAnnealingLR

# T_max = EPOCHS (200)

# EMA: decay = 0.999

# 매 스텝마다 shadow model 업데이트

# 검증 시 .ema_model 사용

# ✅ ========== 학습 제어 ==========

# EarlyStopping 기준: val_logloss

# patience = 20

# 가장 낮은 logloss 모델을 기준으로만 저장

# 저장 경로:

# best_model.pth: base model

# best_model_ema.pth: EMA model

# 로그 저장:

# 파일: train_log.csv

# 컬럼: epoch, train_loss, train_acc, val_loss, val_acc, val_logloss

# ✅ ========== 고정 시드 ==========

# seed = 42

# torch, numpy, random 등 모두 고정

# ✅ ========== 배치 및 학습 파라미터 ==========

# 전체 Epochs: 200

# Batch Size: 32

# Train/Val split: 80% / 20%

# Num workers: 4

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
import random
import numpy as np
import copy

# ✅ SAM Optimizer 정의
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups for p in group['params']
            if p.grad is not None
        ]), p=2)

# ✅ EMA 정의
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

# ✅ 고정 시드
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ✅ 하이퍼파라미터 및 경로 설정
DATA_DIR = '/home/project/car_classification/data/train88'
NUM_CLASSES = 396
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
VAL_RATIO = 0.2
PATIENCE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 데이터셋 & 증강
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ✅ 모델 정의 및 EMA 생성
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-06),
    nn.Dropout(p=0.3),
    nn.Linear(768, NUM_CLASSES)
)
model.to(DEVICE)
ema = EMA(model, decay=0.999)

# ✅ Optimizer & Scheduler & Loss
base_optimizer = AdamW
optimizer = SAM(model.parameters(), base_optimizer, lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=EPOCHS)
criterion_ce = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction='batchmean')

# ✅ 로그 및 얼리스탑
log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_logloss'])
best_val_logloss = float('inf')
early_stop_counter = 0

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        lam = torch.distributions.Beta(0.4, 0.4).sample().item()
        rand_index = torch.randperm(images.size(0)).to(DEVICE)
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        targets_a, targets_b = labels, labels[rand_index]

        # First forward
        outputs1 = model(mixed_images)
        loss_ce1 = lam * criterion_ce(outputs1, targets_a) + (1 - lam) * criterion_ce(outputs1, targets_b)

        outputs2 = model(mixed_images)
        loss_ce2 = lam * criterion_ce(outputs2, targets_a) + (1 - lam) * criterion_ce(outputs2, targets_b)

        log_probs1 = F.log_softmax(outputs1, dim=1)
        log_probs2 = F.log_softmax(outputs2, dim=1)
        kl_loss = (criterion_kl(log_probs1, log_probs2.exp()) + criterion_kl(log_probs2, log_probs1.exp())) / 2

        total_loss = (loss_ce1 + loss_ce2) / 2 + kl_loss * 0.5
        total_loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward for SAM
        outputs1 = model(mixed_images)
        loss_ce1 = lam * criterion_ce(outputs1, targets_a) + (1 - lam) * criterion_ce(outputs1, targets_b)
        loss_ce1.backward()
        optimizer.second_step(zero_grad=True)

        ema.update(model)

        train_loss += total_loss.item() * images.size(0)
        _, predicted = outputs1.max(1)
        train_correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss /= total
    train_acc = train_correct / total * 100

    # ✅ 검증 루프 (EMA 모델 사용)
    ema.ema_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = ema.ema_model(images)
            loss = criterion_ce(outputs, labels)
            val_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(labels.cpu().numpy())
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total * 100
    val_logloss = log_loss(val_targets, val_probs, labels=list(range(NUM_CLASSES)))

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val LogLoss={val_logloss:.6f}")

    log_df.loc[epoch] = [epoch+1, train_loss, train_acc, val_loss, val_acc, val_logloss]

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        torch.save(ema.state_dict(), "best_model_ema.pth")
        print("Best model saved.")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

log_df.to_csv("train_log.csv", index=False)
