import pandas as pd
import numpy as np


normal  = pd.read_csv('press_data_normal.csv', index_col=0)
outlier = pd.read_csv('outlier_data.csv', index_col=0)

normal  = normal.drop_duplicates(keep='first')

print('정상 행:', len(normal), '| 중복:', normal.duplicated().sum())
print('이상 행:', len(outlier), '| 중복:', outlier.duplicated().sum())
print(normal.head(3))

# 1) 정상 데이터 전류 해상도를 이상 데이터(1.192093 단위)에 맞춤
Q = 1.192093036
normal['AI2_Current'] = (np.round(normal['AI2_Current'] / Q) * Q).round(6)

# 2) 두 데이터에 0.2초 기준 구간 번호 부여
for df in (normal, outlier):
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    gap = df['TimeStamp'].diff().dt.total_seconds()
    df['Segment'] = (gap > 0.2).cumsum()

# 3) 결과 확인
for name, df in [('정상', normal), ('이상', outlier)]:
    L = df.groupby('Segment').size()
    print(name, '| 구간 수:', L.size, '| 구간 길이 중앙값:', L.median())
print(normal.iloc[38:44])

# 4) CSV로 저장 (타임스탬프는 원본과 같은 형식으로)
for df, fname in [(normal, 'normal_segmented.csv'), (outlier, 'outlier_segmented.csv')]:
    out = df.copy()
    out['TimeStamp'] = out['TimeStamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    out.to_csv(fname, index_label='OrigIndex')
    print('저장 완료:', fname, len(out), '행')