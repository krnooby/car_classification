import os
import cv2

# ======== 설정 ========
LABEL_DIR = "/home/project/car_classification/data/yoolo"  # YOLO 라벨 + 이미지가 있는 폴더
IMG_DIR = LABEL_DIR
SAVE_DIR = "/home/project/car_classification/data/train_plate_fixed"  # 저장 경로
os.makedirs(SAVE_DIR, exist_ok=True)

# 이미지 확장자 정의
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# YOLO 라벨을 픽셀 좌표로 변환
def yolo_to_pixel_coords(yolo_line, img_w, img_h):
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        return None
    _, x, y, w, h = map(float, parts)
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2

# 전체 이미지 처리
for fname in os.listdir(IMG_DIR):
    if not any(fname.endswith(ext) for ext in IMG_EXTS):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    label_path = os.path.join(LABEL_DIR, fname.rsplit(".", 1)[0] + ".txt")

    image = cv2.imread(img_path)
    if image is None:
        print(f"이미지 로딩 실패: {fname}")
        continue

    h, w = image.shape[:2]

    # 라벨 존재하면 번호판 덮기
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                coords = yolo_to_pixel_coords(line, w, h)
                if coords:
                    x1, y1, x2, y2 = coords
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)  # 흰 박스
    else:
        print(f"[경고] 라벨 없음: {label_path}")

    # 결과 저장
    save_path = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(save_path, image)

print("✅ 번호판 통일 작업 완료! 저장 위치:", SAVE_DIR)
