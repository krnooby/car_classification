import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.models import convnext_small
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

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
# 설정값
# =======================
TEST_DIR = '/home/project/car_classification/data/test'
CSV_PATH = '/home/project/car_classification/data/test.csv'
WEIGHT_PATH = '/home/project/car_classification/outputs/best_model.pth'
SUBMIT_PATH = 'submission.csv'
TRAIN_DIR = '/home/project/car_classification/data/train'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =======================
# 클래스 목록 불러오기
# =======================
raw_class_names = [k for k, v in sorted(ImageFolder(TRAIN_DIR).class_to_idx.items(), key=lambda x: x[1])]
CLASS_NAMES = raw_class_names
NUM_CLASSES = len(raw_class_names)

# =======================
# 기본 전처리 함수
# =======================
def get_base_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# =======================
# 테스트 데이터셋 클래스
# =======================
class TestDataset(Dataset):
    def __init__(self, dataframe, root_dir):
        self.df = dataframe
        self.root_dir = root_dir
        self.img_names = dataframe['img_path'].apply(lambda x: os.path.basename(x)).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 이미지 로딩 실패: {img_path}, 에러: {e}")
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        return image, self.df.iloc[idx]['ID']

# =======================
# 모델 정의 및 로딩 (ConvNeXt-Small)
# =======================
model = convnext_small(weights=None)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Dropout(p=0.3),
    nn.Linear(768, NUM_CLASSES)
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =======================
# TTA 설정
# =======================
TTA_TRANSFORMS = [
    get_base_transform(),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

# =======================
# 추론 수행
# =======================
test_df = pd.read_csv(CSV_PATH)
test_dataset = TestDataset(test_df, TEST_DIR)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))
)

def apply_tta(model, images):
    batch_probs = torch.zeros(len(images), NUM_CLASSES).to(DEVICE)
    for tta_transform in TTA_TRANSFORMS:
        tta_imgs = torch.stack([tta_transform(img) for img in images]).to(DEVICE)
        outputs = model(tta_imgs)
        probs = torch.softmax(outputs, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        batch_probs += probs
    return (batch_probs / len(TTA_TRANSFORMS)).cpu().numpy()

all_probs = []
ids = []

with torch.no_grad():
    for images, id_batch in tqdm(test_loader, desc="Inferencing (TTA x3 avg)"):
        batch_probs = apply_tta(model, images)
        all_probs.extend(batch_probs)
        ids.extend(id_batch)

# =======================
# 결과 저장
# =======================
submission_df = pd.DataFrame(all_probs, columns=CLASS_NAMES)
submission_df.insert(0, 'ID', ids)
submission_df['ID'] = submission_df['ID'].astype(str)
submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
submission_df.to_csv(SUBMIT_PATH, index=False, encoding='utf-8', float_format='%.8f')
print(f"✅ '{SUBMIT_PATH}' 저장 완료!")
