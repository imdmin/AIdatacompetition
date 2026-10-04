from pathlib import Path
import numpy as np
import pandas as pd

# ===== 경로: 이 코드 파일의 위치를 기준으로 찾음 (어디서 실행해도 동작) =====
HERE = Path(__file__).resolve().parent              # 이 코드가 있는 폴더
ROOT = HERE.parent                                   # 프로젝트 폴더
PREP = ROOT / '데이터 전처리 파일'

# ===== 분할 설정 =====
RATIO_NORMAL = (0.6, 0.2, 0.2)    # 정상(A, B 각각): 학습/검증/테스트
RATIO_ANOM   = (0.0, 0.5, 0.5)    # 이상: 학습에는 사용하지 않음
STATE_B      = (15630, 18678)     # 운전 상태 B의 원본 인덱스 범위 (03 운전 상태 분석)

# 1) 전처리 결과 불러오기
normal  = pd.read_csv(PREP / 'prep_normal.csv',  encoding='utf-8-sig', parse_dates=['시작 시각'])
outlier = pd.read_csv(PREP / 'prep_outlier.csv', encoding='utf-8-sig', parse_dates=['시작 시각'])

# 2) 운전 상태 부여
normal['State']  = np.where(normal['시작 원본인덱스'].between(*STATE_B), 'B', 'A')
outlier['State'] = 'anom'
normal['구간키']  = 'N_' + normal['Segment'].astype(str)    # 정상/이상의 구간 번호가 겹치지 않게
outlier['구간키'] = 'X_' + outlier['Segment'].astype(str)
win = pd.concat([normal, outlier], ignore_index=True)
win.insert(0, 'WindowID', range(len(win)))

# 확인: 한 구간 안에서 상태가 섞이지 않는가
mixed = win.groupby('구간키')['State'].nunique()
assert (mixed == 1).all(), '한 구간에 여러 상태가 섞여 있습니다'

# 3) 상태별로 시간순 분할 (구간 단위, 윈도 수 누적 비율 기준)
def split_by_time(df, ratio):
    seg = (df.groupby('구간키')
             .agg(윈도수=('WindowID', 'size'), 시작=('시작 시각', 'min'))
             .sort_values('시작'))
    cum_end = seg['윈도수'].cumsum() / seg['윈도수'].sum()
    b1, b2 = ratio[0], ratio[0] + ratio[1]
    seg['Split'] = np.where(cum_end <= b1 + 1e-9, 'train',
                   np.where(cum_end <= b2 + 1e-9, 'val', 'test'))
    if ratio[0] == 0:
        seg.loc[seg['Split'] == 'train', 'Split'] = 'val'
    return df['구간키'].map(seg['Split'])

win['Split'] = None
for state, ratio in [('A', RATIO_NORMAL), ('B', RATIO_NORMAL), ('anom', RATIO_ANOM)]:
    m = win['State'] == state
    win.loc[m, 'Split'] = split_by_time(win[m], ratio).values

# 4) 결과 확인
order = ['train', 'val', 'test']
summary = (win.groupby(['State', 'Split'])
              .agg(윈도수=('WindowID', 'size'), 구간수=('구간키', 'nunique'),
                   시작시각=('시작 시각', 'min'), 끝시각=('시작 시각', 'max'))
              .reindex(order, level=1))
ratio_tab = (pd.crosstab(win['State'], win['Split'], normalize='index')
               .reindex(columns=order, fill_value=0).round(3))
count_tab = pd.crosstab(win['Split'], win['State']).reindex(order).fillna(0).astype(int)

pd.set_option('display.width', 150)
print('[상태별·용도별 윈도 수, 구간 수, 시간 범위]'); print(summary)
print('\n[상태별 용도 비율]'); print(ratio_tab)
print('\n[용도별 구성]'); print(count_tab)

# 5) 저장
summary.to_csv(HERE / 'table08a_데이터분할_요약.csv', encoding='utf-8-sig')
win.to_csv(HERE / 'windows_split.csv', index=False, encoding='utf-8-sig')
print('\n저장 완료:', HERE / 'windows_split.csv', win.shape)