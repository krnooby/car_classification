import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# ======================= 설정 =======================
TEST_DIR = '/home/project/car_classification/data/test'
CSV_PATH = '/home/project/car_classification/data/test.csv'
WEIGHT_PATH = '/home/project/car_classification/outputs/best_model.pth'
SUBMIT_PATH = 'submission.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 396

# ======================= 전처리 (TTA용 여러 변형) =======================
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

hflip_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TTA_TRANSFORMS = [base_transform, hflip_transform]

# ======================= 모델 정의 및 로드 =======================
def load_model(weight_path):
    model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
        nn.Dropout(p=0.3),
        nn.Linear(768, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ======================= 추론 함수 =======================
def inference():
    model = load_model(WEIGHT_PATH)
    test_df = pd.read_csv(CSV_PATH)
    image_paths = [os.path.basename(p) for p in test_df['img_path'].tolist()]

    all_probs = []
    with torch.no_grad():
        for path in tqdm(image_paths):
            image = Image.open(os.path.join(TEST_DIR, path)).convert('RGB')
            tta_probs = []
            for t in TTA_TRANSFORMS:
                img = t(image).unsqueeze(0).to(DEVICE)
                output = model(img)
                prob = torch.softmax(output, dim=1).cpu().numpy()
                tta_probs.append(prob)
            avg_prob = np.mean(tta_probs, axis=0)
            all_probs.append(avg_prob.squeeze())

    # submission 파일 생성
    sample_sub = pd.read_csv('/home/project/sample_submission.csv')
    submission = pd.DataFrame(all_probs, columns=sample_sub.columns[1:].tolist())
    submission.insert(0, 'ID', test_df['ID'])
    submission.to_csv(SUBMIT_PATH, index=False, encoding='utf-8')
    print(f"✅ '{SUBMIT_PATH}' 저장 완료!")

if __name__ == '__main__':
    inference()
