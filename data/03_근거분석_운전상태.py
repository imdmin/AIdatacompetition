import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

normal = pd.read_csv('normal_segmented.csv', index_col='OrigIndex')
normal['TimeStamp'] = pd.to_datetime(normal['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')

# 구간별 크기 계산
rows = []
for s, g in normal.groupby('Segment'):
    rows.append({'Segment': s, '시작 시각': g['TimeStamp'].iloc[0],
                 '시작 원본인덱스': g.index[0], '끝 원본인덱스': g.index[-1], '길이': len(g),
                 'AI0_RMS': np.sqrt(np.mean(g['AI0_Vibration']**2)),
                 'AI1_RMS': np.sqrt(np.mean(g['AI1_Vibration']**2)),
                 'AI2_최대절댓값': g['AI2_Current'].abs().max()})
seg = pd.DataFrame(rows)
seg.round(4).to_csv('table03a_구간별_크기.csv', index=False, encoding='utf-8-sig')
print('구간 수:', len(seg))
print(seg[['AI0_RMS', 'AI1_RMS', 'AI2_최대절댓값']].describe().round(3))

feat = [('AI0_RMS', '상부 진동 RMS'), ('AI1_RMS', '하부 진동 RMS'), ('AI2_최대절댓값', '전류 최대 절댓값')]

# ① 시간에 따른 변화
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (c, t) in zip(axes, feat):
    ax.plot(seg['시작 시각'], seg[c], 'o', ms=3)
    ax.set_ylabel(t); ax.grid(alpha=0.3)
axes[0].set_title('정상 데이터 구간별 크기의 시간 변화')
axes[-1].set_xlabel('구간 시작 시각')
plt.tight_layout(); plt.savefig('fig03a_구간별크기_시간변화.png', dpi=150); plt.close()

# ② 분포
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (c, t) in zip(axes, feat):
    ax.hist(seg[c], bins=50, edgecolor='k')
    ax.set_title(t); ax.set_xlabel('값'); ax.set_ylabel('구간 수')
plt.tight_layout(); plt.savefig('fig03b_구간별크기_분포.png', dpi=150); plt.close()

# ================= 2단계: 상태 B 경계 결정 =================
TH_AI1 = 0.07     # 하부 진동 RMS 기준 (히스토그램 골짜기)
TH_AI2 = 125      # 전류 최대 절댓값 기준 (히스토그램 골짜기)

seg['낮음'] = (seg['AI1_RMS'] < TH_AI1) & (seg['AI2_최대절댓값'] < TH_AI2)

# 낮음/높음이 연속으로 이어지는 묶음 찾기
run_id = (seg['낮음'] != seg['낮음'].shift()).cumsum()
runs = seg.groupby(run_id).agg(낮음=('낮음', 'first'), 첫구간=('Segment', 'first'),
                               끝구간=('Segment', 'last'), 구간수=('Segment', 'size'),
                               시작시각=('시작 시각', 'first'), 끝시각=('시작 시각', 'last'),
                               시작원본인덱스=('시작 원본인덱스', 'first'),
                               끝원본인덱스=('끝 원본인덱스', 'last'))
low_runs = runs[runs['낮음']].sort_values('구간수', ascending=False)
print(f'\n[2단계] 낮은 구간: 전체 {len(seg)}개 중 {seg["낮음"].sum()}개')
print('연속으로 이어진 낮은 구간 묶음 (긴 순서 상위 5개)')
print(low_runs.head(5).to_string(index=False))

# 가장 긴 묶음 = 상태 B
b = low_runs.iloc[0]
seg['상태'] = np.where((seg['Segment'] >= b['첫구간']) & (seg['Segment'] <= b['끝구간']), 'B', 'A')
print(f'\n상태 B: 구간 {b["첫구간"]}~{b["끝구간"]}, 원본 인덱스 {b["시작원본인덱스"]}~{b["끝원본인덱스"]}, '
      f'{b["시작시각"]} ~ {b["끝시각"]}')

# 검증: B와 A의 낮은 구간이 다른가
seg['비교그룹'] = np.select([seg['상태'] == 'B', seg['낮음']], ['B', 'A-낮은 구간'], 'A-나머지')
comp = seg.groupby('비교그룹')[['AI0_RMS', 'AI1_RMS', 'AI2_최대절댓값', '길이']].median().round(3)
comp['구간 수'] = seg.groupby('비교그룹').size()
print('\n그룹별 중앙값 비교'); print(comp)

low_runs.to_csv('table03b_낮은구간_연속묶음.csv', index=False, encoding='utf-8-sig')
comp.to_csv('table03c_상태별_비교.csv', encoding='utf-8-sig')
seg[['Segment', '시작 원본인덱스', '끝 원본인덱스', '상태']].to_csv('table03d_구간별_상태.csv',
                                                                  index=False, encoding='utf-8-sig')