"""
공통 평가 모듈
모든 모델이 같은 방식으로 확률 보정·기준선 결정을 하고, 같은 지표로 평가받도록 한다.
각 모델 코드는 이 모듈을 불러와 사용한다.

※ 평가 지표는 팀 논의 후 최종 확정 예정. 확정되면 이 파일만 수정하고 모든 모델을 다시 실행한다.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score, matthews_corrcoef,
                             brier_score_loss)

# ===== 공통 설정 =====
SEED = 42          # 난수 고정
N_BOOT = 1000      # 부트스트랩 반복 횟수
WINDOW_SEC = 1.6   # 윈도 1개의 시간 (16샘플 × 0.1초)
RESULT_DIR = Path(__file__).resolve().parent / 'results'

# ===== 모델 선택 규칙 설정 =====
DELTA_F1 = 0.05    # ① F1-score 차이 기준 (검증 이상 윈도 1개의 영향)
TIME_TIE = 0.10    # ③(다른 모델끼리) 추론 시간 차이가 10% 이내면 같은 수준 → 입력 변수가 적은 모델


# ---------- 1) 확률 보정과 기준선 (검증 데이터로 결정) ----------
def fit_calibrator(score_val, y_val, score_ref):
    """
    이상 점수(클수록 이상)를 이상 확률로 바꾸는 보정기를 만든다.
    ① 학습용 정상 데이터의 점수(score_ref)의 평균·표준편차로 점수를 표준화
    ② 표준화한 검증 점수로 로지스틱 회귀 학습
    (표준화는 점수의 순서를 바꾸지 않으므로 판정 결과에는 영향이 없고, 확률의 범위만 바로잡는다)
    """
    ref = np.asarray(score_ref, dtype=float)
    mu, sd = float(ref.mean()), float(ref.std())
    sd = sd if sd > 0 else 1.0
    z_val = (np.asarray(score_val, dtype=float) - mu) / sd
    lr = LogisticRegression().fit(z_val.reshape(-1, 1), y_val)
    return {'mu': mu, 'sd': sd, 'lr': lr}


def to_prob(calibrator, score):
    z = (np.asarray(score, dtype=float) - calibrator['mu']) / calibrator['sd']
    return calibrator['lr'].predict_proba(z.reshape(-1, 1))[:, 1]


def fit_threshold(y_val, p_val):
    """검증 데이터에서 F1-score가 최대인 확률 기준선. 동점이면 높은 기준선(경보가 적은 쪽)"""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1s = (2 * prec * rec / np.maximum(prec + rec, 1e-12))[:-1]
    best = np.flatnonzero(f1s >= f1s.max() - 1e-12)[-1]
    return float(thr[best])


# ---------- 2) 평가 지표 ----------
def evaluate(df, prob, pred):
    """
    df: Label, State, 구간키 열 필요 / prob: 이상 확률 / pred: 0 또는 1
    (윈도당 추론 시간, 학습 시간은 모델 코드에서 측정하여 추가)
    """
    y = df['Label'].to_numpy()
    st = df['State'].to_numpy()
    prob, pred = np.asarray(prob), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    n_anom, n_norm = int((y == 1).sum()), int((y == 0).sum())
    n_a, n_b = int((st == 'A').sum()), int((st == 'B').sum())
    fp_a, fp_b = int(pred[st == 'A'].sum()), int(pred[st == 'B'].sum())
    seg_hit = pd.Series(pred[y == 1]).groupby(df.loc[y == 1, '구간키'].to_numpy()).max()
    normal_hours = n_norm * WINDOW_SEC / 3600
    return {
        # 선택 기준 ①
        'F1-score': f1_score(y, pred, zero_division=0),
        # 오판 지표 (A·B·이상)
        'A 오경보율': fp_a / max(n_a, 1), 'A 오경보 수': fp_a,
        'B 오경보율': fp_b / max(n_b, 1), 'B 오경보 수': fp_b,
        '이상 미탐지율': fn / max(n_anom, 1), '미탐지 수': fn,
        # 종합 성능
        '정밀도': precision_score(y, pred, zero_division=0),
        '재현율': recall_score(y, pred, zero_division=0),
        'MCC': matthews_corrcoef(y, pred),
        'PR-AUC': average_precision_score(y, prob),
        # 확률 품질
        'Brier 점수': brier_score_loss(y, prob),
        # 현장 관점
        '이상 구간 탐지율': float(seg_hit.mean()) if len(seg_hit) else np.nan,
        '시간당 오경보 수': fp / normal_hours if normal_hours > 0 else np.nan,
        # 표본 크기 (해석용)
        '탐지 수': tp, '이상 수': n_anom, 'A 수': n_a, 'B 수': n_b, '이상 구간 수': int(len(seg_hit)),
    }


def measure_inference_ms(predict_fn, df, warmup=2, repeats=20):
    """
    윈도당 추론 시간(ms): 이상 확률을 계산하는 전체 과정의 시간
    준비 실행(warmup)을 먼저 한 뒤 여러 번(repeats) 재서 중앙값 사용 (측정 오차 감소)
    ※ 컴퓨터 성능에 따라 달라지므로, 모델 간 비교는 같은 컴퓨터에서 측정해야 함
    """
    for _ in range(warmup):
        predict_fn(df)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(df)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) / len(df) * 1000)


def boot_ci(df, pred, stat='F1-score'):
    """구간 단위 부트스트랩 95% 신뢰구간 (같은 구간의 윈도는 함께 뽑음)"""
    rng = np.random.default_rng(SEED)
    d = df.assign(_pred=np.asarray(pred))
    groups = {lab: g['구간키'].unique() for lab, g in d.groupby('Label')}
    by_seg = {k: g for k, g in d.groupby('구간키')}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([rng.choice(groups[lab], len(groups[lab]), replace=True) for lab in groups])
        s = pd.concat([by_seg[k] for k in pick])
        if stat == 'F1-score':
            vals.append(f1_score(s['Label'], s['_pred'], zero_division=0))
        else:   # 'B 오경보율'
            b = s[s['State'] == 'B']
            vals.append(b['_pred'].mean() if len(b) else np.nan)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# ---------- 3) 모델 선택 규칙 (검증 결과로만 판단) ----------
def select_best(val_results, n_features, n_b_val, same_algorithm):
    """
    val_results: {모델이름: evaluate() 결과 + '윈도당 추론 시간(ms)'}
    n_features:  {모델이름: 입력 변수 개수}
    same_algorithm: True면 같은 알고리즘끼리 비교 (예: IF_F1 대 IF_F2, 개선 전 대 후)
    ① 검증 F1-score 최고값에서 0.05 이내인 모델만 후보
    ② 후보 중 B 오경보율 최저값에서 B 윈도 1개(1/n_b_val) 이내인 모델만 남김
    ③ 같은 알고리즘끼리: 입력 변수가 가장 적은 모델
       다른 모델끼리: 윈도당 추론 시간이 가장 짧은 모델 (10% 이내 차이는 같은 수준 → 입력 변수가 적은 모델)
    """
    names = list(val_results)
    best_f1 = max(val_results[n]['F1-score'] for n in names)
    c1 = [n for n in names if val_results[n]['F1-score'] >= best_f1 - DELTA_F1 - 1e-9]
    best_fa = min(val_results[n]['B 오경보율'] for n in c1)
    c2 = [n for n in c1 if val_results[n]['B 오경보율'] <= best_fa + 1 / n_b_val + 1e-9]
    if same_algorithm:
        chosen = min(c2, key=lambda n: n_features[n])
        step3 = f'③ 같은 알고리즘 → 입력 변수가 가장 적은 모델: {chosen} ({n_features[chosen]}개)'
    else:
        t_min = min(val_results[n]['윈도당 추론 시간(ms)'] for n in c2)
        c3 = [n for n in c2 if val_results[n]['윈도당 추론 시간(ms)'] <= t_min * (1 + TIME_TIE)]
        chosen = min(c3, key=lambda n: n_features[n])
        step3 = f'③ 다른 모델 → 추론 시간 최저 {t_min:.4f}ms에서 {TIME_TIE:.0%} 이내: {c3} → {chosen}'
    reason = (f'① F1-score 최고 {best_f1:.3f}에서 {DELTA_F1} 이내: {c1} → '
              f'② B 오경보율 최저 {best_fa:.3f}에서 B 윈도 1개({1/n_b_val:.3f}) 이내: {c2} → {step3}')
    return chosen, reason


# ---------- 4) 결과 저장 (모델 비교용 공통 형식) ----------
def save_result(name, info):
    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / f'{name}.json', 'w', encoding='utf-8') as fp:
        json.dump(info, fp, ensure_ascii=False, indent=2, default=float)