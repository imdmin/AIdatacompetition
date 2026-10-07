"""
모델 2: 소형 LSTM-Autoencoder (원신호 3채널)
학습(학습 정상 윈도) → 조기 종료(검증 정상 윈도) → 복원 오차(이상 점수)
→ 확률 보정·기준선(검증 데이터, 공통 평가 모듈) → 검증 평가
※ 테스트 데이터는 모든 모델의 개선·앙상블·최종 선택 후 한 번만 평가한다.
"""
from pathlib import Path
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev                     # 공통 평가 모듈 (같은 폴더)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 경로 =====
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_DIR = ROOT / 'LSTM_원신호 데이터 파일'

# ===== 모델 설정 (데이터 크기와 모델 원리로 미리 정한 값) =====
NAME = 'LSTM_AE'
HIDDEN = 16        # 인코더·디코더 LSTM 크기
LATENT = 8         # 압축 크기
LR = 1e-3          # 학습률 (Adam)
BATCH = 32         # 한 번에 학습하는 윈도 수
MAX_EPOCHS = 1000   # 최대 학습 횟수
PATIENCE = 20      # 검증 정상 오차가 20회 연속 줄지 않으면 조기 종료

# ===== 재현성: 난수 고정 =====
random.seed(ev.SEED)
np.random.seed(ev.SEED)
torch.manual_seed(ev.SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

# 1) 원신호와 윈도 정보 불러오기 (테스트 데이터는 사용하지 않음)
X = np.load(RAW_DIR / 'raw_windows.npy')                         # (윈도 수, 16, 3)
meta = pd.read_csv(RAW_DIR / 'raw_windows_meta.csv', encoding='utf-8-sig')
tr = (meta['Split'] == 'train').to_numpy()
va = (meta['Split'] == 'val').to_numpy()
val_meta = meta[va].reset_index(drop=True)
print('학습:', meta.loc[tr, 'State'].value_counts().to_dict())
print('검증:', val_meta['State'].value_counts().to_dict())

# 2) 스케일링: 변수(채널)마다 학습 정상 데이터의 평균·표준편차로 표준화
mu = X[tr].reshape(-1, X.shape[2]).mean(axis=0)
sd = X[tr].reshape(-1, X.shape[2]).std(axis=0)
sd[sd == 0] = 1.0


def scale(a):
    return torch.tensor(((a - mu) / sd).astype(np.float32))


X_tr = scale(X[tr])
X_va = scale(X[va])
X_va_norm = X_va[torch.tensor(val_meta['Label'].to_numpy() == 0)]   # 조기 종료용: 검증 정상만


# 3) 모델 정의: 인코더 LSTM → 압축 → 디코더 LSTM → 복원
class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):                          # x: (윈도 수, 16, 변수 수)
        _, (h, _) = self.encoder(x)                # 16시점을 읽은 뒤의 마지막 기억
        z = self.to_latent(h[-1])                  # 압축된 요약
        d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)   # 요약을 16시점에 전달
        y, _ = self.decoder(d)
        return self.output(y)                      # 복원된 윈도


model = LSTMAE(n_features=X.shape[2])
n_params = sum(p.numel() for p in model.parameters())
print(f'\n모델 파라미터 수: {n_params}')

# 4) 학습 (학습 정상 윈도) + 조기 종료 (검증 정상 윈도)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr), batch_size=BATCH,
                                     shuffle=True, generator=torch.Generator().manual_seed(ev.SEED))
best_loss, best_state, best_epoch, wait, history = np.inf, None, 0, 0, []
t0 = time.perf_counter()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total = 0.0
    for (xb,) in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), xb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_va_norm), X_va_norm).item()
    history.append({'epoch': epoch, '학습 오차': total / len(X_tr), '검증 정상 오차': val_loss})
    if val_loss < best_loss - 1e-6:
        best_loss, best_epoch, wait = val_loss, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
    if wait >= PATIENCE:
        break
train_sec = time.perf_counter() - t0
model.load_state_dict(best_state)                  # 가장 좋았던 시점으로 되돌림
print(f'학습 종료: {epoch}회 (가장 좋았던 시점 {best_epoch}회, 검증 정상 오차 {best_loss:.4f}), {train_sec:.1f}초')


# 5) 이상 점수 = 윈도의 복원 오차 (48개 값의 평균 제곱 오차)
def recon_error(xs):
    model.eval()
    with torch.no_grad():
        return ((model(xs) - xs) ** 2).mean(dim=(1, 2)).numpy()


s_train = recon_error(X_tr)                        # 학습 정상 오차 (확률 보정의 표준화 기준)
s_val = recon_error(X_va)

# 6) 확률 보정 → 기준선 → 검증 평가 (공통 평가 모듈)
cal = ev.fit_calibrator(s_val, val_meta['Label'], s_train)
p_val = ev.to_prob(cal, s_val)
th = ev.fit_threshold(val_meta['Label'], p_val)
yhat = (p_val >= th).astype(int)

X_va_raw = X[va]


def predict_prob(df):                              # 추론 전 과정: 스케일링 → 복원 → 오차 → 확률
    return ev.to_prob(cal, recon_error(scale(X_va_raw)))


res = ev.evaluate(val_meta, p_val, yhat)
res['윈도당 추론 시간(ms)'] = ev.measure_inference_ms(predict_prob, val_meta)
res['학습 시간(초)'] = train_sec

pd.set_option('display.width', 220)
print('\n[검증 데이터 결과]')
print(pd.DataFrame({NAME: res}).T.round(4).to_string())

# 7) 저장
torch.save(model.state_dict(), HERE / f'model_{NAME}.pt')
joblib.dump({'mu': mu, 'sd': sd, 'calibrator': cal, 'threshold': th,
             'config': {'hidden': HIDDEN, 'latent': LATENT, 'n_features': int(X.shape[2])}},
            HERE / f'model_{NAME}_meta.joblib')
val_meta[['WindowID', 'State', '구간키', 'Label']].assign(
    이상점수=s_val.round(6), 이상확률=p_val.round(4), 예측=yhat
).to_csv(HERE / f'val_predictions_{NAME}.csv', index=False, encoding='utf-8-sig')

hist = pd.DataFrame(history)
hist.to_csv(HERE / f'loss_history_{NAME}.csv', index=False, encoding='utf-8-sig')
plt.figure(figsize=(8, 4))
plt.plot(hist['epoch'], hist['학습 오차'], label='학습 오차')
plt.plot(hist['epoch'], hist['검증 정상 오차'], label='검증 정상 오차')
plt.axvline(best_epoch, color='gray', ls='--', label=f'가장 좋았던 시점 ({best_epoch}회)')
plt.xlabel('학습 횟수'); plt.ylabel('복원 오차 (평균 제곱 오차)'); plt.legend()
plt.title('LSTM-Autoencoder 학습 곡선')
plt.tight_layout(); plt.savefig(HERE / f'fig_loss_{NAME}.png', dpi=130); plt.close()

ev.save_result(NAME, {'모델': '소형 LSTM-Autoencoder', '입력': '원신호 3채널 (16시점 × 3변수)',
                      '입력 변수 개수': int(X.shape[1] * X.shape[2]), '파라미터 수': n_params,
                      '학습 종료 시점': epoch, '가장 좋았던 시점': best_epoch,
                      '기준선': th, '검증': res})
print('\n저장 완료: results 폴더(검증 결과), 검증 예측 파일, 학습 곡선, 모델 파일')