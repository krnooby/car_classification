import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import random

# =======================
# 설정
# =======================
TEST_DIR = '/home/project/car_classification/data/test'
CSV_PATH = '/home/project/car_classification/data/test.csv'
SAMPLE_SUB_PATH = '/home/project/sample_submission.csv'
WEIGHT_PATH = '/home/project/car_classification/outputs/best_model.pth'
TRAIN_DIR = '/home/project/car_classification/data/train'
SUBMIT_PATH = 'submission.csv'
BATCH_SIZE = 32
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
# 클래스 이름 (391개 추론용 + 396개 제출용)
# =======================
dataset_for_class = ImageFolder(TRAIN_DIR)
CLASS_NAMES_391 = [k for k, v in sorted(dataset_for_class.class_to_idx.items(), key=lambda x: x[1])][:391]
NUM_CLASSES = len(CLASS_NAMES_391)

submission_format = pd.read_csv(SAMPLE_SUB_PATH)
SUBMIT_CLASS_NAMES = list(submission_format.columns[1:])  # ID 제외한 클래스 이름들
TOTAL_SUBMIT_CLASSES = len(SUBMIT_CLASS_NAMES)

# =======================
# TTA 전처리 정의
# =======================
TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
]

# =======================
# 테스트 데이터셋 정의
# =======================
class TestDataset(Dataset):
    def __init__(self, dataframe, root_dir):
        self.img_names = dataframe['img_path'].apply(lambda x: os.path.basename(x)).tolist()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        return image, img_name.replace('.jpg', '')

# =======================
# 모델 정의 및 로딩
# =======================
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
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
# TTA 적용 추론 함수
# =======================
def apply_tta(model, images):
    probs_sum = torch.zeros(len(images), NUM_CLASSES).to(DEVICE)
    for transform in TTA_TRANSFORMS:
        inputs = torch.stack([transform(img) for img in images]).to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            probs_sum += probs
    return (probs_sum / len(TTA_TRANSFORMS)).cpu().numpy()

# =======================
# 추론 실행
# =======================
test_df = pd.read_csv(CSV_PATH)
dataset = TestDataset(test_df, TEST_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda x: list(zip(*x)))

all_ids, all_probs = [], []

for images, ids in tqdm(loader, desc="Inferencing with TTA"):
    probs = apply_tta(model, images)
    all_probs.extend(probs)
    all_ids.extend(ids)

# =======================
# 391 → 396 클래스 변환 + 클래스 정렬
# =======================
probs_391 = np.array(all_probs)
dummy_count = TOTAL_SUBMIT_CLASSES - NUM_CLASSES
dummy_probs = np.full((probs_391.shape[0], dummy_count), 1e-7)
probs_396 = np.concatenate([probs_391, dummy_probs], axis=1)
probs_396 /= probs_396.sum(axis=1, keepdims=True)

# 클래스 이름 대응 (name: index)
class_name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES_391)}
dummy_names = [name for name in SUBMIT_CLASS_NAMES if name not in CLASS_NAMES_391]
final_probs = []

for row in probs_396:
    reordered = []
    for name in SUBMIT_CLASS_NAMES:
        if name in class_name_to_idx:
            reordered.append(row[class_name_to_idx[name]])
        else:
            reordered.append(1e-7)  # dummy class 확률 삽입
    final_probs.append(reordered)

# =======================
# 제출 파일 생성 및 저장
# =======================
submit_df = pd.DataFrame(final_probs, columns=SUBMIT_CLASS_NAMES)
submit_df.insert(0, 'ID', all_ids)
submit_df = submit_df.sort_values(by='ID').reset_index(drop=True)
submit_df.to_csv(SUBMIT_PATH, index=False, float_format='%.8f', encoding='utf-8')
print(f"✅ 제출 파일 저장 완료 (396 클래스 맞춤) → {SUBMIT_PATH}")
