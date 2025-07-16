import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 396
IMAGE_DIR = "/home/ubuntu/venv/test"
MODEL_PATH = "best_model.pth"
REFERENCE_SUBMISSION = "/home/ubuntu/venv/submission_basic.csv"  # 클래스 이름 추출용 예시 파일
OUTPUT_CSV = "submission_convnext_small.csv"

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 테스트 데이터셋
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# 클래스 이름 추출
reference_df = pd.read_csv(REFERENCE_SUBMISSION)
class_names = reference_df.columns[1:].tolist()  # 'ID' 제외

# 모델 정의 및 로드
model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=768, out_features=NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 데이터로더
test_dataset = TestDataset(IMAGE_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 추론 및 확률 저장
all_probs = []
image_names = []

with torch.no_grad():
    for images, names in tqdm(test_loader, desc="Running Inference"):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        image_names.extend(names)

# 제출파일 생성
def create_submission(probabilities, image_names, output_path, class_names):
    pred_df = pd.DataFrame(probabilities, columns=class_names)
    pred_df.insert(0, 'ID', image_names)
    pred_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 저장
create_submission(all_probs, image_names, OUTPUT_CSV, class_names)
print(f"Inference complete. Results saved to '{OUTPUT_CSV}'")
