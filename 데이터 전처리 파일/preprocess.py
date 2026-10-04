"""
데이터 전처리 모듈
원본 CSV(TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current[, Equipment_state])를
모델 입력용 윈도 특징 표로 변환한다. 학습 데이터와 신규 데이터에 동일하게 적용한다.

사용법 (터미널, 프로젝트 폴더에서):
    .venv\\Scripts\\python.exe "데이터 전처리 파일\\preprocess.py" 입력파일.csv 출력파일.csv
"""
import sys
import numpy as np
import pandas as pd

# ===== 고정 설정값 (학습 데이터 분석으로 결정, 신규 데이터에도 그대로 적용) =====
Q_CURRENT     = 1.192093036   # 전류 측정 단위 (02 해상도 분석)
GAP_SEC       = 0.2           # 연속 구간 경계 기준, 초 (측정 주기 0.1초의 2배, 02 시간 간격 분석)
MAX_FILL      = 1             # 선형 보간할 최대 연속 결측 개수
W             = 16            # 윈도 길이, 샘플 (04 윈도 길이 분석)
STEP          = 16            # 이동 간격 = 윈도 길이 → 겹치지 않음
BAD_TIME_WARN = 0.10          # 시각 오류 행 비율이 이 값을 넘으면 경고
SENSORS       = {'AI0': 'AI0_Vibration', 'AI1': 'AI1_Vibration', 'AI2': 'AI2_Current'}
TS_FORMAT     = '%Y-%m-%d %H:%M:%S.%f'


def load_raw(path):
    """① 원본 CSV 읽기. 첫 열이 이름 없는 인덱스인 경우와 아닌 경우 모두 처리."""
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith('Unnamed') or first == '':
        df = df.set_index(first)
    df.index.name = 'OrigIndex'
    need = ['TimeStamp'] + list(SENSORS.values())
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f'필수 열이 없습니다: {miss}')
    return df


def clean(df):
    """② 중복 제거 ③ 시각 변환, 시각 오류 행 제거 ④ 센서 오류값을 결측으로 표시"""
    n0 = len(df)
    df = df.drop_duplicates(keep='first').copy()
    n_dup = n0 - len(df)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format=TS_FORMAT, errors='coerce')
    n1 = len(df)
    df = df.dropna(subset=['TimeStamp'])
    n_bad_time = n1 - len(df)
    for col in SENSORS.values():
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df = df.sort_values('TimeStamp', kind='stable')
    info = {'원본 행': n0, '중복 제거 행': n_dup, '시각 오류 제거 행': n_bad_time,
            '센서 결측값 개수': int(df[list(SENSORS.values())].isna().sum().sum())}
    return df, info


def check_time_quality(info):
    """③-1 시각 오류 비율 점검: 기준을 넘으면 경고, 모든 행이 오류면 중단"""
    n = info['원본 행'] - info['중복 제거 행']
    bad = info['시각 오류 제거 행']
    ratio = bad / n if n > 0 else 1.0
    if bad == n:
        raise ValueError('모든 행의 시각을 읽을 수 없습니다. '
                         '시각 형식(예: 2022-07-12 00:00:00.019)을 확인하세요. '
                         '엑셀에서 저장한 파일은 시각이 손상되었을 수 있습니다.')
    if ratio > BAD_TIME_WARN:
        print(f'[경고] 시각 오류 행이 {bad}개({ratio:.1%})로 기준({BAD_TIME_WARN:.0%})을 넘습니다. '
              '원본 파일의 시각 형식을 확인하세요.')


def add_segments(df):
    """⑤ 측정 간격이 0.2초 이상인 지점에서 연속 구간 번호 부여
       (샘플이 하나만 빠져도 간격이 0.2초가 되어 구간이 나뉨)"""
    gap = df['TimeStamp'].diff().dt.total_seconds()
    df['Segment'] = (gap >= GAP_SEC - 1e-9).cumsum()   # 1e-9: 소수 계산 오차 보정
    return df


def fill_missing(df):
    """⑥ 구간 안의 짧은 결측(MAX_FILL개 이하 연속)은 앞뒤 값으로 선형 보간,
          더 긴 결측은 제거하고 그 자리에서 구간을 나눔
       ⑦ 전류 해상도 맞춤"""
    cols = list(SENSORS.values())
    was_na = df[cols].isna()

    def fill_short(s):
        isna = s.isna()
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).transform('size')
        short = isna & (run_len <= MAX_FILL)              # 짧은 결측만 보간 대상
        filled = s.interpolate(method='linear', limit_area='inside')
        return s.where(~short, filled)

    for col in cols:
        df[col] = df.groupby('Segment')[col].transform(fill_short)
    df['보간'] = (was_na & df[cols].notna()).any(axis=1)
    n_filled = int(df['보간'].sum())

    still_na = df[cols].isna().any(axis=1)                # 보간하지 못한 긴 결측
    n_drop = int(still_na.sum())
    df['Segment'] = (df['Segment'].astype(str) + '_'
                     + still_na.groupby(df['Segment']).cumsum().astype(str))
    df = df[~still_na].copy()
    df['Segment'] = pd.factorize(df['Segment'])[0]       # 긴 결측 자리에서 구간 분리

    df['AI2_Current'] = (np.round(df['AI2_Current'] / Q_CURRENT) * Q_CURRENT).round(6)
    return df, {'보간한 행': n_filled, '보간 불가로 제거한 행': n_drop, '정제 후 행': len(df)}


def _features(w):
    """⑨ 윈도 하나(16행)의 특징 31개 계산"""
    r = {}
    for short, col in SENSORS.items():
        x = w[col].to_numpy(dtype=float)
        rms = np.sqrt(np.mean(x ** 2))
        sd = x.std()
        z = (x - x.mean()) / sd if sd > 0 else np.full(len(x), np.nan)
        r[short + '_mean'] = x.mean()
        r[short + '_std'] = sd
        r[short + '_rms'] = rms
        r[short + '_peak'] = np.abs(x).max()
        r[short + '_p2p'] = x.max() - x.min()
        r[short + '_kurt'] = np.mean(z ** 4) - 3
        r[short + '_skew'] = np.mean(z ** 3)
        r[short + '_crest'] = np.abs(x).max() / rms if rms > 0 else np.nan
    X = w[list(SENSORS.values())].to_numpy(dtype=float).T
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.corrcoef(X)
        x2 = X[2]
        r['corr_AI0_AI1'], r['corr_AI0_AI2'], r['corr_AI1_AI2'] = c[0, 1], c[0, 2], c[1, 2]
        r['AI2_period8'] = np.corrcoef(x2[:-8], x2[8:])[0, 1]
    r['ratio_AI0_AI2'] = r['AI0_rms'] / r['AI2_rms'] if r['AI2_rms'] > 0 else np.nan
    r['ratio_AI1_AI2'] = r['AI1_rms'] / r['AI2_rms'] if r['AI2_rms'] > 0 else np.nan
    r['ratio_AI0_AI1'] = r['AI0_rms'] / r['AI1_rms'] if r['AI1_rms'] > 0 else np.nan
    return r


def make_windows(df):
    """⑧ 구간 안에서 16행씩 겹치지 않게 윈도 생성 후 특징 계산"""
    rows = []
    for seg_id, g in df.groupby('Segment'):
        for start in range(0, len(g) - W + 1, STEP):
            w = g.iloc[start:start + W]
            r = {'Segment': seg_id, '시작 원본인덱스': w.index[0],
                 '시작 시각': w['TimeStamp'].iloc[0],
                 '보간포함': bool(w['보간'].any())}
            if 'Equipment_state' in w.columns:
                r['Label'] = int(w['Equipment_state'].iloc[0])
            r.update(_features(w))
            rows.append(r)
    return pd.DataFrame(rows)


def preprocess_file(path):
    """원본 CSV 하나를 윈도 특징 표로 변환 (①~⑩ 전체 과정)"""
    df = load_raw(path)
    df, info = clean(df)
    check_time_quality(info)
    df = add_segments(df)
    df, info2 = fill_missing(df)
    info.update(info2)
    win = make_windows(df)
    meta = ['Segment', '시작 원본인덱스', '시작 시각', 'Label', '보간포함']
    feat_cols = [c for c in win.columns if c not in meta]
    win['판정가능'] = ~win[feat_cols].isna().any(axis=1)  # ⑩ 특징 계산 불가 윈도 표시
    L = df.groupby('Segment').size()
    info.update({'구간 수': int(len(L)), f'{W}샘플 미만 구간 수': int((L < W).sum()),
                 '윈도 수': len(win),
                 '보간값 포함 윈도 수': int(win['보간포함'].sum()),
                 '판정 불가 윈도 수 (특징 계산 불가)': int((~win['판정가능']).sum()),
                 '구간 내 간격이 0.1초가 아닌 경우': int(
                     (df.groupby('Segment')['TimeStamp'].diff().dt.total_seconds()
                        .dropna().round(3) != 0.1).sum())})
    return win, info


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('사용법: python preprocess.py 입력파일.csv 출력파일.csv')
        sys.exit(1)
    win, info = preprocess_file(sys.argv[1])
    for k, v in info.items():
        print(f'{k}: {v}')
    win.to_csv(sys.argv[2], index=False, encoding='utf-8-sig')
    print('저장 완료:', sys.argv[2], win.shape)