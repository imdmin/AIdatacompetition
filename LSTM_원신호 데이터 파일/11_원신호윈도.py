"""
11. LSTM-Autoencoder용 원신호 윈도 추출
기존 전처리 모듈(preprocess.py)의 정제·구간 분할·결측 처리 함수를 그대로 사용하여,
Isolation Forest와 '같은 윈도'의 원래 값(16시점 × 3변수)을 꺼내고,
windows_split.csv의 WindowID와 학습·검증·테스트 구분을 그대로 붙인다.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ===== 경로 =====
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / '데이터 전처리 파일'))
import preprocess as pp                       # 기존 전처리 모듈 재사용

RAW = {'N': ROOT / 'data' / 'press_data_normal.csv',     # N: 정상, X: 이상 (구간키 접두어와 같음)
       'X': ROOT / 'data' / 'outlier_data.csv'}
SPLIT_FILE = ROOT / '데이터 분할 파일' / 'windows_split.csv'
CHANNELS = list(pp.SENSORS.values())          # ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']


# 1) 원본 → 기존 전처리 (정제, 구간 분할, 결측 처리, 전류 해상도 맞춤)
def preprocess_raw(path):
    df = pp.load_raw(path)
    df, info = pp.clean(df)
    pp.check_time_quality(info)
    df = pp.add_segments(df)
    df, _ = pp.fill_missing(df)
    return df


# 2) 기존과 같은 규칙으로 윈도를 자르고, 특징 대신 원래 값(16 × 3)을 저장
def raw_windows(df, prefix):
    keys, arrays = [], []
    for seg_id, g in df.groupby('Segment'):
        for start in range(0, len(g) - pp.W + 1, pp.STEP):
            w = g.iloc[start:start + pp.W]
            keys.append((f'{prefix}_{seg_id}', int(w.index[0])))      # (구간키, 시작 원본인덱스)
            arrays.append(w[CHANNELS].to_numpy(dtype=np.float32))
    return keys, arrays


key_to_array = {}
for prefix, path in RAW.items():
    keys, arrays = raw_windows(preprocess_raw(path), prefix)
    key_to_array.update(dict(zip(keys, arrays)))
    print(f'{path.name}: 윈도 {len(keys)}개 추출')

# 3) windows_split.csv의 순서·WindowID·Split에 맞춰 정렬
split = pd.read_csv(SPLIT_FILE, encoding='utf-8-sig')
split = split[split['판정가능']].reset_index(drop=True)
pairs = list(zip(split['구간키'], split['시작 원본인덱스'].astype(int)))
missing = [p for p in pairs if p not in key_to_array]
assert not missing, f'분할 파일의 윈도 {len(missing)}개를 원신호에서 찾지 못함: {missing[:5]}'
X = np.stack([key_to_array[p] for p in pairs])                  # (윈도 수, 16, 3)

# 4) 검증: 원신호로 다시 계산한 RMS가 분할 파일의 RMS와 같은지 (같은 윈도인지 확인)
rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=1))          # (윈도 수, 3)
diff = np.abs(rms - split[['AI0_rms', 'AI1_rms', 'AI2_rms']].to_numpy()).max()
print(f'\n원신호 배열 크기: {X.shape} (윈도 수, 시점, 변수)')
print(f'RMS 일치 확인: 최대 차이 {diff:.2e} → {"일치" if diff < 1e-4 else "불일치 (확인 필요)"}')
print('\n[용도별 윈도 수]')
print(pd.crosstab(split['Split'], split['State']).reindex(['train', 'val', 'test']))

# 5) 저장: 원신호 배열 + 윈도 정보(같은 순서)
np.save(HERE / 'raw_windows.npy', X)
split[['WindowID', 'Split', 'State', 'Label', '구간키', '시작 원본인덱스', '시작 시각']].to_csv(
    HERE / 'raw_windows_meta.csv', index=False, encoding='utf-8-sig')
print('\n저장 완료: raw_windows.npy (원신호), raw_windows_meta.csv (윈도 정보, 같은 순서)')