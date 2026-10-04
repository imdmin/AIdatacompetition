"""
모델 1: 시계열 특징 + Isolation Forest
학습(학습 데이터) → 이상 점수 → 확률 보정·기준선(검증 데이터) → 검증 평가 → F1/F2 선택
※ 테스트 데이터는 모든 모델의 개선·앙상블·최종 선택 후 한 번만 평가한다.
"""
from pathlib import Path
import json
import time
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
import evaluation as ev                     # 공통 평가 모듈 (같은 폴더)

# ===== 경로 =====
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLIT_FILE = ROOT / '데이터 분할 파일' / 'windows_split.csv'
FEAT_FILE = ROOT / '특징 선택 파일' / 'selected_features.json'

# ===== 모델 설정 =====
N_TREES = 300

# 1) 데이터와 특징 불러오기 (테스트 데이터는 사용하지 않음)
win = pd.read_csv(SPLIT_FILE, encoding='utf-8-sig')
win = win[win['판정가능']].copy()
with open(FEAT_FILE, encoding='utf-8') as fp:
    FEATS = json.load(fp)
SETS = {'IF_F1': FEATS['F1'], 'IF_F2': FEATS['F2']}
train = win[win['Split'] == 'train']
val = win[win['Split'] == 'val']
print('학습:', train['State'].value_counts().to_dict())
print('검증:', val['State'].value_counts().to_dict())

# 2) 학습 → 이상 점수 → 확률 보정 → 기준선 → 검증 평가
bundles, val_res = {}, {}
for name, cols in SETS.items():
    t0 = time.perf_counter()
    iso = IsolationForest(n_estimators=N_TREES, random_state=ev.SEED).fit(train[cols])
    train_sec = time.perf_counter() - t0

    s_train = -iso.score_samples(train[cols])                    # 학습 정상 데이터의 점수 (표준화 기준)
    s_val = -iso.score_samples(val[cols])                        # 이상 점수: 클수록 이상
    cal = ev.fit_calibrator(s_val, val['Label'], s_train)
    p_val = ev.to_prob(cal, s_val)
    th = ev.fit_threshold(val['Label'], p_val)
    yhat = (p_val >= th).astype(int)

    def predict_prob(df, iso=iso, cal=cal, cols=cols):          # 추론 전 과정 (점수 → 확률)
        return ev.to_prob(cal, -iso.score_samples(df[cols]))

    r = ev.evaluate(val, p_val, yhat)
    r['윈도당 추론 시간(ms)'] = ev.measure_inference_ms(predict_prob, val)
    r['학습 시간(초)'] = train_sec
    val_res[name] = r

    bundles[name] = {'model': iso, 'calibrator': cal, 'threshold': th, 'features': cols}
    joblib.dump(bundles[name], HERE / f'model_{name}.joblib')
    val[['WindowID', 'State', '구간키', 'Label']].assign(
        이상점수=s_val.round(5), 이상확률=p_val.round(4), 예측=yhat
    ).to_csv(HERE / f'val_predictions_{name}.csv', index=False, encoding='utf-8-sig')

pd.set_option('display.width', 220)
print('\n[검증 데이터 결과]')
print(pd.DataFrame(val_res).T.round(4).to_string())

# 3) IF 안에서 F1/F2 선택 (검증 결과만, 같은 알고리즘 → ③은 입력 변수 개수)
n_features = {n: len(c) for n, c in SETS.items()}
n_b_val = int((val['State'] == 'B').sum())
chosen, reason = ev.select_best(val_res, n_features, n_b_val, same_algorithm=True)
print(f'\n[IF 안에서의 선택] {chosen}\n  {reason}')

# 4) 결과 저장 (모델 비교용 공통 형식)
for name in SETS:
    ev.save_result(name, {'모델': 'Isolation Forest', '특징': bundles[name]['features'],
                          '기준선': bundles[name]['threshold'], '검증': val_res[name],
                          '모델 내 선택': name == chosen, '선택 이유': reason})
print('\n저장 완료: results 폴더(검증 결과), 검증 예측 파일, 모델 파일')