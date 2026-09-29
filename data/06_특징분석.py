import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

win = pd.read_csv('windows_16.csv', encoding='utf-8-sig')
FEATS = [c for c in win.columns
         if c not in ['WindowID', 'Label', 'State', 'Segment', '시작 원본인덱스', '시작 시각']]
print('특징 개수:', len(FEATS))

# ① 상태별 중앙값
med = win.groupby('State')[FEATS].median().T[['A', 'B', 'anom']]
med.round(4).to_csv('table06a_상태별_중앙값.csv', encoding='utf-8-sig')

# ② 구분력(AUC): 특징 하나로 두 그룹을 얼마나 잘 나누나 (0.5=못 나눔, 1.0=완벽)
def auc(g1, g2, f):
    d = win[win['State'].isin([g1, g2])]
    a = roc_auc_score((d['State'] == g1).astype(int), d[f])
    return max(a, 1 - a)

rows = []
for f in FEATS:
    rows.append({'특징': f, '이상 vs A': auc('anom', 'A', f),
                 '이상 vs B': auc('anom', 'B', f), 'B vs A': auc('B', 'A', f)})
sep = pd.DataFrame(rows).set_index('특징').round(3).sort_values('이상 vs A', ascending=False)
sep.to_csv('table06b_특징별_구분력.csv', encoding='utf-8-sig')
print('\n[② 구분력 AUC] 이상 vs A 순서')
print(sep.to_string())

# ① 상태별 분포 상자그림
fig, axes = plt.subplots(7, 4, figsize=(16, 24))
for ax, f in zip(axes.ravel(), FEATS):
    data = [win.loc[win['State'] == s, f] for s in ['A', 'B', 'anom']]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2, 3], ['정상A', '정상B', '이상'])
    ax.set_title(f)
plt.tight_layout(); plt.savefig('fig06a_특징별_상태분포.png', dpi=120); plt.close()

# ③ 특징끼리 겹치는 정보 (정상 데이터만 사용)
corr = win.loc[win['Label'] == 0, FEATS].corr(method='spearman')
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATS) for b in FEATS[i + 1:]
         if abs(corr.loc[a, b]) >= 0.9]
red = pd.DataFrame(pairs, columns=['특징1', '특징2', '순위상관']).round(3)
red.to_csv('table06c_중복특징_쌍.csv', index=False, encoding='utf-8-sig')
print('\n[③ 중복] 순위상관 |r| >= 0.9 인 특징 쌍 (정상 데이터 기준)')
print(red.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90)
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS)
plt.colorbar(im); ax.set_title('특징 간 순위상관 (정상 데이터)')
plt.tight_layout(); plt.savefig('fig06b_특징간_상관.png', dpi=120); plt.close()