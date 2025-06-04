# ==========사용된 모델===== best code.
# ConvNeXt-Base

# ==========데이터 증강=====
# RandomResizedCrop(224, scale=(0.9, 1.0)): 거의 전체 이미지를 유지하면서 약간의 crop
# RandomHorizontalFlip(): 좌우 반전
# RandAugment(): 자동 조정된 복합 augmentation
# Normalize(): ImageNet 평균/표준편차 기준 정규화

# ==========데이터 전처리 (val)==============================
# Resize(256) → CenterCrop(224): 검증 데이터 정규화

# ==========학습 전략=====================================
# MixUp & CutMix 병행 (확률적으로 한 쪽 선택)
# alpha=0.3로 soft label 구성
# KLDivLoss + softmax(log): soft target에 맞춘 KL loss
# Label Smoothing 포함된 CrossEntropyLoss: 검증용

# ==========최적화 및 정규화===============================
# Optimizer: AdamW (weight_decay=1e-4)
# Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
# EMA (Exponential Moving Average): 모델 파라미터 평균을 따로 추적하여 검증에 사용

# ========== 학습 제어=======================================
# EarlyStopping: patience=10
# → val loss 또는 val logloss가 향상되지 않으면 중단
# Best Checkpoint 저장: /outputs/best_model.pth
# Epoch Checkpoint 저장: /outputs/ckpt_epochXX.pth
# =============================================================

# ======================= 라이브러리 =======================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
import random
import numpy as np
import copy
from torchvision.datasets import ImageFolder

# ======================= 설정 =======================
PATIENCE = 10
DATA_DIR = '/home/project/car_classification/data/train'
BATCH_SIZE = 32
EPOCHS = 300
LR = 5e-5
VAL_RATIO = 0.2
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIX_ALPHA = 0.3
EMA_DECAY = 0.999
RESUME_PATH = None  # 예: '/home/project/car_classification/outputs/ckpt_epoch25.pth'

# ======================= 시드 고정 =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ======================= 데이터 전처리 =======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(DATA_DIR, transform=train_transform)
NUM_CLASSES = len(dataset.classes)

val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ======================= 모델 정의 =======================
model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((1024,), eps=1e-06, elementwise_affine=True),
    nn.Dropout(p=0.4),
    nn.Linear(1024, NUM_CLASSES)
)
if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"\n📦 Resuming from checkpoint: {RESUME_PATH}")
    model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
model.to(DEVICE)
ema_model = copy.deepcopy(model)

# ======================= 손실 함수 및 최적화 설정 =======================
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
kl_loss = nn.KLDivLoss(reduction="batchmean")

# ======================= MixUp & CutMix =======================
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mix_or_cut(images, labels, alpha=MIX_ALPHA):
    index = torch.randperm(images.size(0)).to(images.device)
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        images = lam * images + (1 - lam) * images[index]
        labels = lam * F.one_hot(labels, NUM_CLASSES).float() + (1 - lam) * F.one_hot(labels[index], NUM_CLASSES).float()
    else:
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels = lam_adj * F.one_hot(labels, NUM_CLASSES).float() + (1 - lam_adj) * F.one_hot(labels[index], NUM_CLASSES).float()
    return images, labels

# ======================= 학습 루프 =======================
log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
best_val_logloss = float('inf')
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, targets = mix_or_cut(images, labels)
        outputs = model(images)
        loss = kl_loss(F.log_softmax(outputs, dim=1), targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(targets.argmax(dim=1)).sum().item()
        total += labels.size(0)

    scheduler.step(epoch)
    train_loss /= total
    train_acc = train_correct / total * 100

    # ======================= 검증 =======================
    ema_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_probs, val_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = ema_model(images)
            loss = ce_loss(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            val_correct += outputs.argmax(1).eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total * 100
    val_logloss = log_loss(val_targets, val_probs, labels=list(range(NUM_CLASSES)))
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val LogLoss={val_logloss:.6f}")
    log_df.loc[epoch] = [epoch+1, train_loss, train_acc, val_loss, val_acc]

    # ======================= 체크포인트 및 얼리스탑 =======================
    if (val_logloss < best_val_logloss) or (val_loss < best_val_loss):
        checkpoint_path = f"/home/project/car_classification/outputs/ckpt_epoch{epoch+1:02d}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        if val_logloss < best_val_logloss: best_val_logloss = val_logloss
        if val_loss < best_val_loss: best_val_loss = val_loss

        early_stop_counter = 0
        os.makedirs('/home/project/car_classification/outputs', exist_ok=True)
        torch.save(ema_model.state_dict(), "/home/project/car_classification/outputs/best_model.pth")
        print("Best model saved.")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# ======================= 로그 저장 =======================
log_df.to_csv("train_log.csv", index=False)
