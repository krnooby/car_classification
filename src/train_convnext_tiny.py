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

# =======================
# 하이퍼파라미터 설정
# =======================
PATIENCE = 7
DATA_DIR = '/home/project/car_classification/data/train'
NUM_CLASSES = 396
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
# SAM Optimizer
# =======================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
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
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

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

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =======================
# 모델 정의 (ConvNeXt-Tiny)
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
base_optimizer = AdamW
optimizer = SAM(model.parameters(), base_optimizer, lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=EPOCHS)
criterion_ce = nn.CrossEntropyLoss()

# =======================
# 체크포인트 이어서 학습
# =======================
CHECKPOINT_DIR = '/home/project/car_classification/outputs/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

START_EPOCH = 0
last_checkpoint_path = ''
for i in reversed(range(1, EPOCHS + 1)):
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{i}.pth')
    if os.path.exists(path):
        last_checkpoint_path = path
        break

if last_checkpoint_path:
    checkpoint = torch.load(last_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print(f"Loaded checkpoint from {last_checkpoint_path}, resuming from epoch {START_EPOCH}")
else:
    print("No checkpoint found. Starting from scratch.")

# =======================
# 학습 루프
# =======================
log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
best_val_logloss = float('inf')
early_stop_counter = 0

for epoch in range(START_EPOCH, EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion_ce(outputs, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs = model(images)
        loss = criterion_ce(outputs, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss /= total
    train_acc = train_correct / total * 100

    # 검증
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

    log_df.loc[epoch] = [epoch+1, train_loss, train_acc, val_loss, val_acc]

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        early_stop_counter = 0
        os.makedirs('/home/project/car_classification/outputs', exist_ok=True)
        torch.save(model.state_dict(), "/home/project/car_classification/outputs/best_model.pth")
        print("Best model (logloss) saved.")
    else:
        early_stop_counter += 1

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_logloss': val_logloss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# =======================
# 로그 저장
# =======================
log_df.to_csv("train_log.csv", index=False)
