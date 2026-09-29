import pandas as pd
import numpy as np

W = 16        # 윈도 길이 (04에서 확정)
STEP = 16     # 이동 간격 = 윈도 길이 → 겹치지 않게

# 1) 불러오기
normal  = pd.read_csv('normal_segmented.csv',  index_col='OrigIndex')
outlier = pd.read_csv('outlier_segmented.csv', index_col='OrigIndex')
state   = pd.read_csv('table03d_구간별_상태.csv', encoding='utf-8-sig')
for df in (normal, outlier):
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')

# 2) 상태 붙이기 (03의 결과 파일 사용)
normal['State'] = normal['Segment'].map(state.set_index('Segment')['상태'])
outlier['State'] = 'anom'

COLS = {'AI0': 'AI0_Vibration', 'AI1': 'AI1_Vibration', 'AI2': 'AI2_Current'}

# 3) 구간 안에서 윈도 만들기 + 특징 계산
def make_windows(df, label):
    rows = []
    for seg_id, g in df.groupby('Segment'):
        for start in range(0, len(g) - W + 1, STEP):
            w = g.iloc[start:start + W]
            r = {'Label': label, 'State': w['State'].iloc[0], 'Segment': seg_id,
                 '시작 원본인덱스': w.index[0], '시작 시각': w['TimeStamp'].iloc[0]}
            for short, col in COLS.items():
                x = w[col].to_numpy()
                rms = np.sqrt(np.mean(x**2))
                sd = x.std()
                z = (x - x.mean()) / sd if sd > 0 else np.full(W, np.nan)
                r[short + '_mean']  = x.mean()
                r[short + '_std']   = sd
                r[short + '_rms']   = rms
                r[short + '_peak']  = np.abs(x).max()
                r[short + '_p2p']   = x.max() - x.min()
                r[short + '_kurt']  = np.mean(z**4) - 3
                r[short + '_skew']  = np.mean(z**3)
                r[short + '_crest'] = np.abs(x).max() / rms
            c = np.corrcoef(w[list(COLS.values())].to_numpy().T)
            r['corr_AI0_AI1'] = c[0, 1]
            r['corr_AI0_AI2'] = c[0, 2]
            r['corr_AI1_AI2'] = c[1, 2]
            x2 = w['AI2_Current'].to_numpy()
            r['AI2_period8'] = np.corrcoef(x2[:-8], x2[8:])[0, 1]
            rows.append(r)
    return pd.DataFrame(rows)

win = pd.concat([make_windows(normal, 0), make_windows(outlier, 1)], ignore_index=True)
win.insert(0, 'WindowID', range(len(win)))

# 4) 확인 및 저장
print(win.groupby('State').agg(윈도수=('WindowID', 'size'), 구간수=('Segment', 'nunique')))
print('결측값 개수:', int(win.isna().sum().sum()))
win.to_csv('windows_16.csv', index=False, encoding='utf-8-sig')
print('저장 완료: windows_16.csv', win.shape)