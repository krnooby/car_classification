import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from torchvision.datasets import ImageFolder

# =======================
# 하이퍼파라미터 설정
# =======================
PATIENCE = 7
DATA_DIR = '/home/project/car_classification/data/train'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =======================
# 시드 고정
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =======================
# 클래스 병합 정보
# =======================
MERGE_CLASSES = [
    ('K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'),
    ('디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'),
    ('718_박스터_2017_2024', '박스터_718_2017_2024'),
    ('RAV4_2016_2018', '라브4_4세대_2013_2018'),
    ('RAV4_5세대_2019_2024', '라브4_5세대_2019_2024'),
]

def create_merged_class_mapping(original_classes):
    merged_name_map = {}
    used = set()
    for group in MERGE_CLASSES:
        merged = sorted(group)[0]
        for name in group:
            merged_name_map[name] = merged
            used.add(name)
    for name in original_classes:
        if name not in used:
            merged_name_map[name] = name
    return merged_name_map

class ImageFolderWithMerge(ImageFolder):
    def __init__(self, root, transform=None, merged_name_map=None):
        super().__init__(root, transform=transform)
        self.class_to_merged = merged_name_map or {}
        new_classes = sorted(set(self.class_to_merged[c] for c in self.classes))
        self.class_to_idx = {c: i for i, c in enumerate(new_classes)}
        new_samples = []
        for path, orig_idx in self.samples:
            orig_class = self.classes[orig_idx]
            merged_class = self.class_to_merged[orig_class]
            merged_idx = self.class_to_idx[merged_class]
            new_samples.append((path, merged_idx))
        self.samples = new_samples
        self.targets = [s[1] for s in new_samples]
        self.classes = new_classes

# =======================
# 데이터 전처리
# =======================
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

original_classes = os.listdir(DATA_DIR)
merged_name_map = create_merged_class_mapping(original_classes)
dataset = ImageFolderWithMerge(root=DATA_DIR, transform=transform, merged_name_map=merged_name_map)
NUM_CLASSES = len(dataset.classes)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =======================
# 모델 정의
# =======================
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Dropout(p=0.3),
    nn.Linear(768, NUM_CLASSES)
)
model.to(DEVICE)

# =======================
# Optimizer, Scheduler, Loss
# =======================
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# =======================
# 학습 루프
# =======================
log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
best_val_logloss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss /= total
    train_acc = train_correct / total * 100

    # =======================
    # 검증
    # =======================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
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

    log_df.loc[epoch] = [epoch+1, train_loss, train_acc, val_loss, val_acc]

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        early_stop_counter = 0
        os.makedirs('/home/project/car_classification/outputs', exist_ok=True)
        torch.save(model.state_dict(), "/home/project/car_classification/outputs/best_model.pth")
        print("Best model (logloss) saved.")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# =======================
# 로그 저장
# =======================
log_df.to_csv("train_log.csv", index=False)
