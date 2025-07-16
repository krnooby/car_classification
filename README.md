# 🚗 중고차 차종 분류 모델 - ConvNeXt Tiny 기반

본 프로젝트는 중고차 이미지를 활용해 **396개 차종을 분류**하는 딥러닝 모델을 구축하며,  
최적화된 학습 기법과 강력한 추론 전략을 통해 **리더보드 상위권 진입**을 목표로 합니다.

---

## 🔍 프로젝트 주요 정보

| 항목 | 내용 |
|------|------|
| 기반 모델 | ConvNeXt-Tiny (`torchvision`) |
| 프레임워크 | PyTorch |
| 입력 이미지 | RGB, 224x224 |
| 출력 클래스 | 396개 차종 |
| 사전학습 | ImageNet-1k |
| 성능 향상 전략 | MixUp, R-Drop, SAM, EMA, TTA |
| 최종 점수 | `0.1989` (logloss 기준) |

---

## 📦 모델 구조

- 모델: `ConvNeXt-Tiny`
- classifier 커스터마이징:
  ```python
  nn.Sequential(
      nn.Flatten(),
      nn.LayerNorm((768,), eps=1e-06),
      nn.Dropout(p=0.3),
      nn.Linear(768, 396)
  )
  ```

- 사전학습 가중치:
  ```python
  convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
  ```

---

## 🏗️ 학습 구성 (`convnext_tiny.py`)

### 💡 데이터 증강
- `RandomResizedCrop`, `HorizontalFlip`, `Rotation`, `ColorJitter`
- `Affine`, `Perspective`, `Normalize`

### ⚙️ 학습 전략
- `MixUp(Beta(0.4, 0.4))`: 입력과 라벨 혼합
- `R-Drop`: 두 번 forward + KLDivLoss
- `SAM + AdamW`: 평탄한 최적점 탐색
- `CosineAnnealingLR`: 학습률 조정
- `EMA`: 매 step마다 shadow model 업데이트

### 🧪 Early Stopping
- 기준 지표: `val_logloss`
- patience: `20`
- `best_model.pth`, `best_model_ema.pth` 저장

### 📊 로그 저장
- `train_log.csv` 자동 생성
- 컬럼: `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_logloss`

---

## 🔍 추론 구성 (`inference.py`)

### 💻 모델 로딩
- EMA 가중치 사용: `best_model_ema.pth`

### 🔁 TTA (Test Time Augmentation)
총 3종류 transform을 사용하여 예측값 평균:
1. 기본 Resize
2. Horizontal Flip
3. Random Rotation(15도)

### 📂 결과 저장
- 파일명: `convnexttiny_ema_submission.csv`
- 구성: `ID + 396 클래스별 확률`

---

## 🛠️ 실행 방법

### 📌 학습
```bash
python convnext_tiny.py
```

### 📌 추론
```bash
python inference.py
```

- `test.csv`, `sample_submission.csv`는 `data/` 디렉토리 내 존재해야 함

---

## 💻 실행 환경

| 항목 | 권장 사양 |
|------|----------|
| GPU | RTX 4070 이상 |
| VRAM | 12GB 이상 |
| 학습 시간 | 약 3~6시간 (200 epoch) |
| 프레임워크 | PyTorch >= 1.13, torchvision >= 0.14 |

---

## 📈 결과 예시

| Metric | Value |
|--------|--------|
| Best Val Accuracy | 약 97.4% |
| Best LogLoss | **0.1989** |
| Final Score | 상위 3% 이내 예상 |

---

## ✍️ 주요 기술 요약

- ✅ `ConvNeXt-Tiny`: Efficient + 정확도 균형 우수한 최신 CNN
- ✅ `MixUp + R-Drop`: 일반화 강화
- ✅ `SAM`: 평탄한 손실 지형 유도
- ✅ `EMA`: 테스트 안정성 확보
- ✅ `TTA`: 추론 강건성 향상
