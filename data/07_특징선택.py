import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

AUC_TH = 0.7                                  # ① 구분력 기준
CORR_TH = 0.9                                 # ② 중복 기준
ALWAYS = ['AI0_rms', 'AI1_rms', 'AI2_rms']    # ③ 관계용으로 항상 유지

win = pd.read_csv('windows_16.csv', encoding='utf-8-sig')
META = ['WindowID', 'Label', 'State', 'Segment', '시작 원본인덱스', '시작 시각']

# 관계를 담은 비율 특징 추가 (방법 가)
win['ratio_AI0_AI2'] = win['AI0_rms'] / win['AI2_rms']   # 전류 대비 상부 진동
win['ratio_AI1_AI2'] = win['AI1_rms'] / win['AI2_rms']   # 전류 대비 하부 진동
win['ratio_AI0_AI1'] = win['AI0_rms'] / win['AI1_rms']   # 하부 대비 상부 진동
FEATS = [c for c in win.columns if c not in META]

def sensor(f):
    if f.startswith(('corr_', 'ratio_')):
        return '관계'
    return f.split('_')[0]

def auc(df, g1, g2, f):
    d = df[df['State'].isin([g1, g2])]
    a = roc_auc_score((d['State'] == g1).astype(int), d[f])
    return max(a, 1 - a)

def select_features(df, feats, auc_th=AUC_TH, corr_th=CORR_TH, always=ALWAYS):
    # ① 구분력
    tab = pd.DataFrame({'센서': [sensor(f) for f in feats],
                        '이상 vs A': [auc(df, 'anom', 'A', f) for f in feats],
                        '이상 vs B': [auc(df, 'anom', 'B', f) for f in feats],
                        'B vs A(참고)': [auc(df, 'B', 'A', f) for f in feats]}, index=feats)
    tab['최소 구분력'] = tab[['이상 vs A', '이상 vs B']].min(axis=1)
    tab['① 통과'] = tab['최소 구분력'] >= auc_th
    passed = tab[tab['① 통과']].sort_values('최소 구분력', ascending=False).index.tolist()

    # ② 중복 정리: 같은 센서끼리만, 정상 데이터 순위상관 기준
    corr = df.loc[df['Label'] == 0, passed].corr(method='spearman').abs()
    group, gid = {}, 0
    for f in passed:
        if f in group:
            continue
        gid += 1
        stack = [f]
        while stack:
            x = stack.pop()
            if x in group:
                continue
            group[x] = gid
            stack += [y for y in passed if y not in group
                      and sensor(y) == sensor(x) and corr.loc[x, y] >= corr_th]
    tab['② 묶음 번호'] = pd.Series(group)
    rep = tab.loc[passed].groupby('② 묶음 번호')['최소 구분력'].idxmax()
    tab['② 대표 선택'] = tab.index.isin(rep.values)

    # ③ 관계용 유지
    tab['③ 관계용 유지'] = tab.index.isin(always)
    tab['최종 선택'] = tab['② 대표 선택'] | tab['③ 관계용 유지']
    return tab.round(3)

# 미리보기 (전체 데이터)
tab = select_features(win, FEATS).sort_values(['최종 선택', '최소 구분력'], ascending=[False, False])
tab.to_csv('table07a_특징선택_미리보기.csv', encoding='utf-8-sig')
print('[특징 선택 미리보기 - 전체 데이터]')
print(tab.to_string())
print('\n① 통과:', int(tab['① 통과'].sum()), '개')
print('최종 선택:', tab.index[tab['최종 선택']].tolist())

# ③의 근거 확인: 세 센서 크기의 관계가 상태별로 어떻게 다른가
amp = ['AI0_rms', 'AI1_rms', 'AI2_rms']
rows = []
for name in ['A', 'B', 'anom']:
    c = win.loc[win['State'] == name, amp].corr(method='spearman')
    rows.append({'그룹': name, '윈도 수': int((win['State'] == name).sum()),
                 '상부진동-하부진동': c.loc['AI0_rms', 'AI1_rms'],
                 '상부진동-전류': c.loc['AI0_rms', 'AI2_rms'],
                 '하부진동-전류': c.loc['AI1_rms', 'AI2_rms']})
rel = pd.DataFrame(rows).round(3)
rel.to_csv('table07b_상태별_크기관계.csv', index=False, encoding='utf-8-sig')
print('\n[관계 확인] 상태별 크기 특징 간 순위상관')
print(rel.to_string(index=False))