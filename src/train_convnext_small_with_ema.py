import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
import random
import numpy as np
import copy
from torchvision.datasets import ImageFolder

# ======================= 설정 =======================
PATIENCE = 7
DATA_DIR = '/home/project/car_classification/data/train'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIXUP_ALPHA = 0.4
EMA_DECAY = 0.999

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

dataset = ImageFolder(DATA_DIR, transform=transform)
NUM_CLASSES = len(dataset.classes)

val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ======================= 모델 =======================
model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Dropout(p=0.3),
    nn.Linear(768, NUM_CLASSES)
)
model.to(DEVICE)

ema_model = copy.deepcopy(model)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
kl_loss = nn.KLDivLoss(reduction="batchmean")

# ======================= MixUp 함수 =======================
def mixup(images, labels, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index, :]
    labels_onehot = F.one_hot(labels, NUM_CLASSES).float()
    labels_shuffled = F.one_hot(labels[index], NUM_CLASSES).float()
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled
    return mixed_images, mixed_labels

# ======================= 학습 루프 =======================
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
        images, targets = mixup(images, labels)
        outputs = model(images)
        loss = kl_loss(F.log_softmax(outputs, dim=1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA 업데이트
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        hard_labels = targets.argmax(dim=1)
        train_correct += predicted.eq(hard_labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss /= total
    train_acc = train_correct / total * 100

    # ======================= 검증 =======================
    ema_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = ema_model(images)
            loss = ce_loss(outputs, labels)
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
        torch.save(ema_model.state_dict(), "/home/project/car_classification/outputs/best_model.pth")
        print("Best model (logloss) saved.")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# ======================= 로그 저장 =======================
log_df.to_csv("train_log.csv", index=False)
