import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'      # 한글 글꼴
plt.rcParams['axes.unicode_minus'] = False

bins   = [-np.inf, 0.0999, 0.1001, 0.2, 1, 5, 10, np.inf]
labels = ['0.1초 미만', '0.1초', '0.1~0.2초', '0.2~1초', '1~5초', '5~10초', '10초 이상']
table = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, fname) in zip(axes, [('정상', 'press_data_normal.csv'), ('이상', 'outlier_data.csv')]):
    df = pd.read_csv(fname, index_col=0).drop_duplicates(keep='first')
    ts = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dt = ts.diff().dt.total_seconds().dropna().round(3)     # 연속한 두 행의 시간 차

    # 근거 ①
    n01 = (dt == 0.1).sum()
    print(f'[{name}] 전체 간격 {len(dt)}개 중 0.1초: {n01}개 ({n01 / len(dt) * 100:.2f}%)')
    other = dt[dt != 0.1]
    print(f'   0.1초가 아닌 간격: {len(other)}개 (최소 {other.min()}초, 최대 {other.max()}초)')

    # 근거 ②
    counts = pd.cut(dt, bins=bins, labels=labels).value_counts().reindex(labels)
    table[name] = counts
    ax.bar(labels, counts.values)
    ax.set_yscale('log')
    ax.set_title(f'{name} 데이터 시간 간격 분포')
    ax.set_ylabel('간격 개수 (로그 척도)')
    ax.tick_params(axis='x', rotation=45)

    # 근거 ③
    for th in [0.15, 0.2, 0.5, 1.0]:
        print(f'   기준 {th}초 → 구간 수 {(dt > th).sum() + 1}개')

pd.DataFrame(table).to_csv('table02_시간간격분포.csv', encoding='utf-8-sig')
plt.tight_layout()
plt.savefig('fig02_시간간격분포.png', dpi=150)
print(pd.DataFrame(table))


# ================= 해상도 분석 =================
raw = {'정상': pd.read_csv('press_data_normal.csv', index_col=0).drop_duplicates(keep='first'),
       '이상': pd.read_csv('outlier_data.csv', index_col=0)}
cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

# 근거 ①: 변수별 측정 정밀도 (서로 다른 값들 사이의 최소 간격)
rows = []
for name, df in raw.items():
    for c in cols:
        u = np.sort(df[c].unique())
        d = np.diff(u)
        rows.append({'데이터': name, '변수': c, '전체 값 개수': len(df),
                     '서로 다른 값 개수': len(u), '최소 간격': d.min(), '간격 중앙값': np.median(d)})
res = pd.DataFrame(rows)
res.to_csv('table02b_해상도_최소간격.csv', index=False, encoding='utf-8-sig')
print('\n[해상도 ①] 변수별 측정값의 최소 간격')
print(res.to_string(index=False))

# 근거 ②: 이상 전류가 일정한 단위 q의 정수배인가
x = raw['이상']['AI2_Current']
q0 = np.diff(np.sort(x.unique())).min()          # 최소 간격을 단위 후보로
k = np.round(x / q0)
q = (x * k).sum() / (k * k).sum()                # 모든 값에 가장 잘 맞도록 단위를 보정
print(f'\n[해상도 ②] 단위 후보(최소 간격) = {q0:.9f}, 보정된 단위 q = {q:.9f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, name in zip(axes, ['정상', '이상']):
    v = raw[name]['AI2_Current']
    frac = v / q - np.round(v / q)                # q로 나눴을 때의 소수 부분 (-0.5 ~ 0.5)
    print(f'   {name} 전류: q의 정수배인 값 {100 * (np.abs(frac) < 1e-3).mean():.1f}%, '
          f'소수 부분 최대 {np.abs(frac).max():.4f}')
    ax.hist(frac, bins=50, edgecolor='k')
    ax.set_title(f'{name} 데이터 전류값 ÷ q 의 소수 부분')
    ax.set_xlabel('소수 부분 (0이면 q의 정수배)'); ax.set_ylabel('개수')
plt.tight_layout(); plt.savefig('fig02b_전류해상도.png', dpi=150); plt.close()

# 근거 ③: 정상 전류를 q 단위로 반올림하면 얼마나 바뀌는가
n = raw['정상']['AI2_Current']
nq = np.round(n / q) * q
print(f'\n[해상도 ③] 정상 전류를 q 단위로 반올림했을 때')
print(f'   값의 변화: 최대 {np.abs(nq - n).max():.4f}, 평균 {np.abs(nq - n).mean():.4f}')
print(f'   정상 전류의 표준편차: {n.std():.3f} → 반올림 후 {nq.std():.3f}')
print(f'   서로 다른 값 개수: {n.nunique()} → 반올림 후 {nq.nunique()}')

# 보완: 측정 단위를 신호 크기와 비교 (최소 간격 ÷ 표준편차)
res['표준편차'] = [raw[r['데이터']][r['변수']].std() for _, r in res.iterrows()]
res['최소간격÷표준편차(%)'] = 100 * res['최소 간격'] / res['표준편차']
res.to_csv('table02b_해상도_최소간격.csv', index=False, encoding='utf-8-sig')
print('\n[해상도 ① 보완] 신호 크기 대비 측정 단위')
print(res[['데이터', '변수', '최소 간격', '표준편차', '최소간격÷표준편차(%)']].to_string(index=False))