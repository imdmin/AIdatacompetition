from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ===== 경로: 이 코드 파일의 위치를 기준으로 찾음 =====
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLIT_FILE = ROOT / '데이터 분할 파일' / 'windows_split.csv'
PREVIEW_FILE = ROOT / 'data' / 'table07a_특징선택_미리보기.csv'   # 07 미리보기 (비교용, 없어도 됨)

# ===== 선택 규칙 (07에서 정한 규칙) =====
AUC_TH = 0.7                                   # ① 구분력 기준
CORR_TH = 0.9                                  # ② 중복 기준
ALWAYS = ['AI0_rms', 'AI1_rms', 'AI2_rms']     # ③ 관계용으로 항상 유지
CURRENT_SHAPE = ['AI2_period8', 'AI2_kurt', 'AI2_crest', 'AI2_skew', 'AI2_mean']  # F2에서 제외할 전류 파형 특징

META = ['WindowID', 'Label', 'State', 'Segment', '구간키', 'Split',
        '시작 원본인덱스', '시작 시각', '보간포함', '판정가능']

# 1) 학습용 + 검증용만 사용 (테스트용은 보지 않음)
win = pd.read_csv(SPLIT_FILE, encoding='utf-8-sig')
dev = win[win['Split'].isin(['train', 'val']) & win['판정가능']].copy()
FEATS = [c for c in win.columns if c not in META]
print(f'특징 후보: {len(FEATS)}개')
print('사용 데이터 (학습+검증):', dev['State'].value_counts().to_dict())

def sensor(f):
    return '관계' if f.startswith(('corr_', 'ratio_')) else f.split('_')[0]

def auc(df, g1, g2, f):
    d = df[df['State'].isin([g1, g2])]
    a = roc_auc_score((d['State'] == g1).astype(int), d[f])
    return max(a, 1 - a)

def select_features(df, feats):
    # ① 구분력: 이상 vs A, 이상 vs B 둘 다 기준 이상
    tab = pd.DataFrame({'센서': [sensor(f) for f in feats],
                        '이상 vs A': [auc(df, 'anom', 'A', f) for f in feats],
                        '이상 vs B': [auc(df, 'anom', 'B', f) for f in feats],
                        'B vs A(참고)': [auc(df, 'B', 'A', f) for f in feats]}, index=feats)
    tab['최소 구분력'] = tab[['이상 vs A', '이상 vs B']].min(axis=1)
    tab['① 통과'] = tab['최소 구분력'] >= AUC_TH
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
                      and sensor(y) == sensor(x) and corr.loc[x, y] >= CORR_TH]
    tab['② 묶음 번호'] = pd.Series(group)
    rep = tab.loc[passed].groupby('② 묶음 번호')['최소 구분력'].idxmax()
    # ③의 특징이 속한 묶음은 그 특징을 대표로 함 (②와 ③의 중복 방지)
    for f in ALWAYS:
        if f in group:
            rep[group[f]] = f
    tab['② 대표 선택'] = tab.index.isin(rep.values)

    # ③ 관계 유지
    tab['③ 관계용 유지'] = tab.index.isin(ALWAYS)
    tab['최종 선택'] = tab['② 대표 선택'] | tab['③ 관계용 유지']
    return tab.round(3)

# 2) 규칙 적용
tab = select_features(dev, FEATS).sort_values(['최종 선택', '최소 구분력'], ascending=[False, False])
F1 = tab.index[tab['최종 선택']].tolist()
F2 = [f for f in F1 if f not in CURRENT_SHAPE]

pd.set_option('display.width', 160)
print('\n[특징 선택 결과 - 학습+검증 데이터]')
print(tab.to_string())
print(f'\n① 통과: {int(tab["① 통과"].sum())}개')
print(f'F1 ({len(F1)}개): {F1}')
print(f'F2 ({len(F2)}개): {F2}')

# 3) 07 미리보기(전체 데이터)와 비교
if PREVIEW_FILE.exists():
    prev = pd.read_csv(PREVIEW_FILE, encoding='utf-8-sig', index_col=0)
    prev_sel = prev.index[prev['최종 선택']].tolist()
    print('\n[07 미리보기와 비교]')
    print('  미리보기 선택:', prev_sel)
    print('  새로 들어온 특징:', [f for f in F1 if f not in prev_sel])
    print('  빠진 특징:', [f for f in prev_sel if f not in F1])
else:
    print('\n(07 미리보기 파일이 없어 비교를 건너뜀)')

# 4) 저장: 모델 코드가 이 파일을 읽어서 특징을 사용
tab.to_csv(HERE / 'table09a_특징선택.csv', encoding='utf-8-sig')
with open(HERE / 'selected_features.json', 'w', encoding='utf-8') as fp:
    json.dump({'F1': F1, 'F2': F2,
               '규칙': {'AUC 기준': AUC_TH, '중복 기준': CORR_TH, '항상 유지': ALWAYS,
                        'F2 제외 특징': CURRENT_SHAPE, '사용 데이터': '학습+검증'}},
              fp, ensure_ascii=False, indent=2)
print('\n저장 완료: table09a_특징선택.csv, selected_features.json')