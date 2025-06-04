import pandas as pd
import numpy as np

# 확률 배열 불러오기 (추론 시 np.save로 저장했다고 가정)
all_probs = np.load('all_probs.npy')  # (8258, 396)

# test.csv에서 ID 목록 추출
test_df = pd.read_csv('test.csv')
ids = test_df['img_path'].apply(lambda x: x.split('/')[-1].replace('.jpg', '')).tolist()

# 클래스 이름 강제 지정
CLASS_NAMES = [f'K5_{i:03}' for i in range(396)]

# 제출 파일 생성
submission_df = pd.DataFrame(all_probs, columns=CLASS_NAMES)
submission_df.insert(0, 'ID', ids)
submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
submission_df.to_csv('submission.csv', index=False, encoding='utf-8', float_format='%.8f')

print("✅ 확률 기반 submission.csv 저장 완료")
