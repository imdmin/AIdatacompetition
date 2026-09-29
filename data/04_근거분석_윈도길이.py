import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                       # 그래프를 창에 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

normal  = pd.read_csv('normal_segmented.csv',  index_col='OrigIndex')
outlier = pd.read_csv('outlier_segmented.csv', index_col='OrigIndex')

# 공통 함수: 구간을 넘지 않는 시차 상관
def seg_acf(df, col, max_lag):
    rows = []
    for lag in range(1, max_lag + 1):
        a, b = [], []
        for _, g in df.groupby('Segment'):
            x = g[col].to_numpy()
            if len(x) > lag:
                a.append(x[:-lag]); b.append(x[lag:])
        a, b = np.concatenate(a), np.concatenate(b)
        rows.append({'시차(샘플)': lag, '시차(초)': lag / 10,
                     '자기상관': np.corrcoef(a, b)[0, 1], '쌍의 개수': len(a)})
    return pd.DataFrame(rows)

# ================= A. 전류 파형 주기 =================
acf_cur = seg_acf(normal, 'AI2_Current', 40)
acf_cur.round(3).to_csv('table04a_전류_자기상관.csv', index=False, encoding='utf-8-sig')
v = acf_cur['자기상관'].to_numpy(); lags = acf_cur['시차(샘플)'].to_numpy()
mins = [int(lags[i]) for i in range(1, len(v) - 1) if v[i] < v[i-1] and v[i] < v[i+1]]
maxs = [int(lags[i]) for i in range(1, len(v) - 1) if v[i] > v[i-1] and v[i] > v[i+1]]
print('[A] 자기상관 골짜기(시차):', mins)
print('    자기상관 봉우리(시차):', maxs)
if len(mins) >= 2:
    print('    한 주기 추정 = 첫 두 골짜기 간격:', mins[1] - mins[0], '샘플')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(acf_cur['시차(샘플)'], acf_cur['자기상관'], 'o-')
axes[0].axhline(0, color='gray', lw=0.8)
axes[0].set_xlabel('시차 (샘플, 1샘플=0.1초)'); axes[0].set_ylabel('자기상관')
axes[0].set_title('정상 데이터 전류의 구간 내 자기상관')
longest = normal.groupby('Segment').size().idxmax()
seg = normal[normal['Segment'] == longest]
axes[1].plot(np.arange(len(seg)), seg['AI2_Current'].to_numpy(), 'o-')
axes[1].set_xlabel('구간 내 샘플 번호'); axes[1].set_ylabel('AI2_Current')
axes[1].set_title(f'정상 데이터 전류 원신호 예시 (구간 {longest})')
plt.tight_layout(); plt.savefig('fig04a_전류주기.png', dpi=150); plt.close()

# ================= B. 큰 진동 지속 시간 =================
THR = normal['AI0_Vibration'].abs().max()      # 가정 1: 정상 데이터에서 한 번도 넘지 않은 크기
MERGE = 3                                       # 가정 2: 3샘플(0.3초) 이내 초과는 같은 진동
print(f'[B] 큰 진동 기준: |AI0| > {THR:.4f} (정상 데이터의 최댓값)')
events = []
for s, g in outlier.groupby('Segment'):
    over = np.where(g['AI0_Vibration'].abs().to_numpy() > THR)[0]
    if len(over) == 0:
        continue
    start = prev = over[0]
    for i in list(over[1:]) + [None]:
        if i is not None and i - prev <= MERGE:
            prev = i; continue
        events.append({'Segment': s, '시작 위치': start, '끝 위치': prev,
                       '지속(샘플)': prev - start + 1,
                       '구간 경계에 걸림': start == 0 or prev == len(g) - 1})
        if i is not None:
            start = prev = i
ev = pd.DataFrame(events)
ev.to_csv('table04b_큰진동_목록.csv', index=False, encoding='utf-8-sig')
print('    큰 진동 개수:', len(ev), '| 구간 경계에 걸려 잘린 것:', int(ev['구간 경계에 걸림'].sum()))
print('    지속 시간(샘플) 요약:'); print(ev['지속(샘플)'].describe().round(1).to_string())
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ev['지속(샘플)'], bins=range(1, ev['지속(샘플)'].max() + 2), edgecolor='k')
ax.set_xlabel('큰 진동 지속 시간 (샘플, 1샘플=0.1초)'); ax.set_ylabel('개수')
ax.set_title('이상 데이터 큰 진동의 지속 시간 분포')
plt.tight_layout(); plt.savefig('fig04b_큰진동지속.png', dpi=150); plt.close()

done = ev[~ev['구간 경계에 걸림']]
print('    잘리지 않은 진동만:'); print(done['지속(샘플)'].describe().round(1).to_string())

# ================= C. 윈도 길이별 데이터 손실 =================
rows = []
for W in range(8,31):
    r = {'윈도 길이': W}
    for name, df in [('정상', normal), ('이상', outlier)]:
        L = df.groupby('Segment').size()
        used = L[L >= W]
        n_win = used // W                                    # 겹치지 않게 자를 때 구간별 윈도 수
        r[f'{name} 사용 구간'] = len(used)
        r[f'{name} 제외 구간 비율(%)'] = round(100 * (1 - used.sum() / len(df)), 1)
        r[f'{name} 윈도 수'] = int(n_win.sum())
        r[f'{name} 실제 미사용(%)'] = round(100 * (1 - (n_win * W).sum() / len(df)), 1)
    rows.append(r)
loss = pd.DataFrame(rows)
loss.to_csv('table04c_윈도길이별_손실.csv', index=False, encoding='utf-8-sig')
print('[C] 윈도 길이별 데이터 손실'); print(loss.to_string(index=False))

# ================= D. 윈도 길이별 위상 의존성 (가상 사인파) =================
# 주기가 P인 사인파를 시작 위치를 200가지로 바꿔가며 자르고, 표준편차가 얼마나 흔들리는지 측정
rows = []
for P in [16, 16.5, 17]:                       # A의 추정 범위(16~17)를 모두 확인
    for W in range(8,31):
        stds = [np.std(np.sin(2 * np.pi * (np.arange(W) + ph) / P))
                for ph in np.linspace(0, P, 200, endpoint=False)]
        rows.append({'가정 주기': P, '윈도 길이': W, '주기 대비 길이': round(W / P, 2),
                     '표준편차 최소': round(min(stds), 3), '표준편차 최대': round(max(stds), 3),
                     '흔들림(%)': round(100 * (max(stds) - min(stds)) / (1 / np.sqrt(2)), 1)})
phase = pd.DataFrame(rows)
phase.to_csv('table04d_윈도길이별_위상의존성.csv', index=False, encoding='utf-8-sig')
print('[D] 윈도 길이별 표준편차 흔들림(%) — 0에 가까울수록 시작 위치와 무관')
print(phase.pivot(index='윈도 길이', columns='가정 주기', values='흔들림(%)'))

# ================= E. 구간 길이 분포 확인 =================
for name, df in [('정상', normal), ('이상', outlier)]:
    L = df.groupby('Segment').size()
    print(f'[E] {name}: 전체 {len(L)}개 구간 중 길이가 정확히 50인 구간 {(L == 50).sum()}개, 최대 길이 {L.max()}')

    # ================= F. 조건 종합 =================
flick = phase.groupby('윈도 길이')['흔들림(%)'].max().rename('최대 흔들림(%)')
summ = loss.set_index('윈도 길이').join(flick)
summ['① 흔들림<10%'] = summ['최대 흔들림(%)'] < 10
summ['② 9샘플 이상'] = summ.index >= 9
summ['③ 이상 구간≥10'] = summ['이상 사용 구간'] >= 10
summ['모두 만족'] = summ[['① 흔들림<10%', '② 9샘플 이상', '③ 이상 구간≥10']].all(axis=1)
cols = ['최대 흔들림(%)', '이상 사용 구간', '정상 윈도 수', '이상 윈도 수',
        '정상 실제 미사용(%)', '이상 실제 미사용(%)', '모두 만족']
summ[cols].to_csv('table04e_조건종합.csv', encoding='utf-8-sig')
print('\n[F] 조건 종합')    
print(summ[cols].to_string())