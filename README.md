## 강화학습 기반 코인 포트폴리오 트레이딩 (RL_Trading_Bot)

본 저장소는 Binance Vision 시세 데이터를 기반으로, 포트폴리오 비중을 학습하는 강화학습 트레이딩 실험 코드입니다. 핵심 알고리즘은 Implicit Q-Learning(IQL)의 가치 추정과 확산 모형(diffusion) 기반 행동 모델을 결합(IDQL)하여, 상태-가치 기반의 평가와 행동 분포 학습을 동시에 수행합니다. 환경은 리스크 민감 보상 함수와 거래비용(슬리피지) 모델을 포함합니다.


---

### 주요 특징
- **데이터 파이프라인**: Binance Vision 월별 CSV를 읽고, 규칙 격자(reindex), 결측/마스크, per-feature 경과시간(elapsed)을 포함한 6개 엔지니어드 특징을 생성
- **환경(`RLTradingEnv`)**: N개 자산 + 현금(USDT) 비중을 액션으로 하는 포트폴리오 환경, 가우시안 거래비용/슬리피지, 리스크 민감 보상
- **보상(`RiskAwareReward`)**: 핵심수익(ret_core), 다운사이드(sigma_down), 벤치마크 대비 차등수익(Dret), Treynor를 조합한 합성 보상 (참고: [관련 논문](https://arxiv.org/pdf/2506.04358))
- **에이전트(`IDQLAgent`)**: IQL(Value expectile + TD Q) + 확산정책(variance-preserving, epsilon 예측), 폴리시 인코더 EMA 분리 (참고: [IDQL 논문](https://arxiv.org/pdf/2304.10573))
- **리플레이 버퍼**: CPU 텐서 기반 고속 샘플링, 핀 메모리로 GPU 이전 최적화
- **검증/로그/체크포인트**: 주기적 평가, CSV 메트릭 기록, 모델 체크포인트 저장/로드

---

### 학습 코드 동작 순서

1) 설정/로깅 초기화
- `configs/train_config.yaml` 로드 → 로깅/검증/체크포인트 설정 읽기
- 실행 로그 디렉터리 생성, 메트릭 레코더 초기화

2) 데이터 로드 및 전처리
- 학습 구간 CSV 로드: `data.binance.load_csv.load_vision_monthlies_csv`
- 전처리 파이프라인 적용: `set_index_and_sort` → `sanity_check` → `ensure_regular_grid` → `drop_outside_features`
- 동일 절차로 검증 구간 데이터 준비

3) 데이터셋/환경 구성
- `DatasetConfig(window_size, base_interval, strict_anchor, symbol_order)` 생성
- `EnvConfig(symbols, min/max_steps, transaction_cost_mu/sigma, reward_window, reward_weights, reward_annualize, seed)` 생성
- `RLTradingEnv` 초기화 (보상·슬리피지·포트폴리오 상태 포함)

4) 디바이스/리플레이 버퍼/하이퍼파라미터 설정
- 디바이스 선택(CUDA/CPU), `ReplayBuffer(capacity, ready_after)` 생성
- 배치크기, 업데이트 주기/반복, 에피소드·총스텝 한도 등 하이퍼파라미터 로드

5) 에이전트(IDQL) 초기화
- `env.reset()` 후 상태를 `pack_state_for_buffer`로 패킹하여 per-step 특징 결합 차원(D_packed) 산출
- `IDQLConfig` 구성 후 `IDQLAgent` 생성(인코더/가치·Q/확산 정책 초기화 및 베타 스케줄 등록)

6) 학습 루프(에피소드 반복)
- 각 에피소드: `env.reset()` → done까지 반복
  - 상태 패킹 → `agent.act(state)`로 액션 산출(확산 정책 후보 샘플 + Q 스코어링, ε-그리디 탐험 포함)
  - `env.step(action)` → 보상/컴포넌트/다음 상태 수신
  - 전이 `(s,a,r,s',done)`를 리플레이 버퍼에 저장, 카운터 갱신
  - 버퍼 준비·스텝 조건 충족 시 업데이트 버스트 실행:
    - `train_iters_per_update` 회 반복: 버퍼 샘플 → 디바이스 전송 → `agent.update(batch)` → 손실/Q/V 통계 누적
- 에피소드 통계 로깅 및 메트릭 기록, 조기종료 조건(`max_total_env_steps`) 검사

7) 주기적 검증 및 체크포인트(옵션)
- `every_episodes` 간격으로 체크포인트 저장(활성 시) 후 `environment.validation.evaluate_agent` 실행
- 에이전트·균등분배·BTC-온리 최종수익 통계를 기록

8) 종료 처리
- 에피소드/검증 메트릭 CSV를 실행 폴더에 저장

- IDQL 업데이트 순서 요약: (1) 인코더+V+Q(IQL) → (2) `q_target` 및 폴리시 인코더 EMA 갱신 → (3) 확산 정책(로짓 공간 BC) 학습

---

### 리포지토리 구조
```
configs/
  └─ train_config.yaml            # 학습/환경/에이전트 설정 파일
src/
  ├─ train.py                     # 메인 학습 엔트리포인트
  ├─ models/IDQL.py               # IQL + Diffusion 정책 에이전트
  ├─ environment/
  │   ├─ tradingEnv.py           # RL 환경 (포트폴리오/슬리피지/보상 연결)
  │   ├─ reward.py               # 리스크 민감 보상 구현
  │   └─ validation.py           # 에이전트/베이스라인 평가 유틸
  ├─ data/binance/
  │   ├─ load_csv.py             # Binance Vision CSV 로딩/머지
  │   ├─ preprocess.py           # 전처리 파이프라인(격자/정합/정리)
  │   ├─ dataset.py              # 특징(6개) + 마스크 + elapsed 구성 Dataset
  │   ├─ constants.py            # 특징/인터벌 상수 및 유틸
  │   ├─ verification.py         # Kline 누락/중복 점검 스크립트
  │   └─ vision.py               # Vision 다운로드/체크섬 유틸
  ├─ utils/
  │   ├─ replay_buffer/replay_buffer.py  # CPU 리플레이 버퍼
  │   └─ logging/train_logging.py        # 로깅/메트릭 CSV 기록
  └─ download_binance_history_data.py    # Vision 데이터 대량 다운로드 예시 스크립트
requirements.txt
logs/, models/                        # 실행 시 생성(로그/체크포인트)
```

---

### 설치
- 요구사항: Python 3.10+, Linux 환경 권장, CUDA(GPU) 선택사항
- 설치:
```bash
pip install -r requirements.txt
```

---

### 데이터 준비
이 프로젝트는 Binance Vision의 폴더 구조를 그대로 사용합니다. `configs/train_config.yaml`의 `binance_kline_dataset_root` 경로가 다음 형태를 가리키도록 맞추세요:

```
<ROOT>/spot/monthly/klines/<SYMBOL>/<INTERVAL>/<SYMBOL>-<INTERVAL>-YYYY-MM.csv
```

- 예: `/.../data/binance/raw/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-07.csv`
- 데이터 다운로드 예시 스크립트:
```bash
python src/download_binance_history_data.py
```
스크립트는 ZIP과 .CHECKSUM을 내려받아 무결성 확인 후 압축을 풀어 폴더에 CSV를 생성합니다. 기본 출력 루트는 `src/download_binance_history_data.py`의 `DOWNLOAD_FOLDER_ROOT`를 확인/수정하세요.

데이터 품질 점검(누락/중복 캔들):
```bash
python -c "from src.data.binance.verification import print_report, scan_symbol_folder; \
import sys; from pathlib import Path; p=Path(sys.argv[1]); print_report(scan_symbol_folder(p))" \
  /path/to/spot/monthly/klines/BTCUSDT
```

---

### 빠른 시작 (학습 실행)
1) 설정 파일 편집: `configs/train_config.yaml`에서 심볼/기간/경로/하이퍼파라미터를 확인하세요.

2) 학습 실행:
```bash
# 프로젝트 루트에서
python -m src.train
# 또는
python src/train.py
```

3) 출력물:
- 로그/메트릭: `logging.root_dir` 하위의 자동 생성 run 폴더에 `train.log`, `training_metrics.csv`, `validation_metrics.csv`
- 체크포인트: `checkpoint.dir` 하위에 에피소드/스텝 기준으로 `.pt` 저장

---

### 구성(config) 설명 (요약)
`configs/train_config.yaml` 주요 키:

- **상위**
  - **binance_kline_dataset_root**: Vision CSV 루트 (`.../spot/monthly/klines`)
  - **symbols**: 학습/검증에 사용할 종목 리스트
  - **train_start_month/train_end_month**: 학습 데이터 월 범위 `YYYY-MM`
  - **preprocessing_pipeline**: 전처리 스텝 리스트 (`set_index_and_sort`, `sanity_check`, `ensure_regular_grid`, `drop_outside_features`)
  - **window_size**: 슬라이딩 윈도우 길이 L
  - **base_interval**: 앵커 기준 인터벌(예: `1m`)
  - **strict_anchor**: 모든 (심볼,인터벌) 조합이 L개 이상 있을 때만 앵커 사용
  - **random_seed**: 시드
  - **slippage_mu/sigma**: 거래비용(단순수익 단위) 가우시안 파라미터

- **env**
  - **min_steps/max_steps**: 에피소드 길이 범위(스텝)
  - **reward_weights**: 보상 가중치 `[w1,w2,w3,w4]`
  - **reward_annualize**: 핵심수익을 연율화할지 여부

- **replay_buffer**
  - **capacity/ready_after**: 버퍼 크기와 업데이트 시작 최소 크기

- **training**
  - **batch_size**: 미니배치 크기
  - **min_steps_for_update_replay**: 업데이트 버스트 간 최소 환경 스텝
  - **train_iters_per_update**: 한 번에 실행할 업데이트 반복 수
  - **max_episodes/max_total_env_steps**: 학습 조기종료 조건

- **idql** (에이전트)
  - **state_emb_dim/hidden**: 인코더/헤드 너비
  - **gamma/tau_expectile/lr/target_ema**: IQL 관련 하이퍼파라미터
  - **diff_T/diff_time_dim/diff_blocks/diff_dropout**: 확산모형 구조
  - **diff_samples_per_state**: 액션 샘플 수(행동 선택 시)
  - **diff_beta_min/max/schedule**: 베타 스케줄
  - **policy_ema**: 크리틱 인코더 → 폴리시 인코더 EMA 비율
  - **epsilon_explore**: 입실론-그리디 탐험 확률(Dirichlet 샘플)

- **validation**
  - **enabled/every_episodes/n_episodes/start_month/end_month**: 검증 주기/횟수/기간

- **checkpoint**
  - **enabled/dir**: 체크포인트 저장 활성화/경로

- **logging**
  - **root_dir/level/run_name**: 로그 루트/레벨/실행명

---

### 데이터셋/특징 생성 요약
- 심볼×인터벌 조합마다 다음 6개 특징을 사용합니다: `r_oc_prev`, `r_co`, `r_ho`, `r_lo`, `v_log1p`, `taker_imbalance`
- 각 특징에 대해 마스크(`__mask`)와 경과시간(`__elapsed`, log1p 분)을 함께 생성하여 총 3C 채널을 구성합니다.
- 윈도우 내 채널별 z-score 정규화(단, `taker_imbalance`는 [-1,1] 유지) 후 NaN→0 치환
- `strict_anchor=true`이면 모든 조합에 대해 충분한 윈도우가 존재하는 타임스탬프만 앵커로 사용됩니다.

---

### 환경/보상 설계
액션은 길이 `N+1`의 단순 비중 벡터(마지막이 현금)이며, 합은 1입니다. 거래비용은 가우시안 샘플을 기반으로 half-turnover 규칙으로 적용됩니다.

보상 합성식(요약):
- 창 길이 `window`에서, `mu_p_net = mean(rp_gross) - mean(cost)`
- `ret_core = annualize ? periods_per_year * mu_p_net : mu_p_net`
- `sigma_down = sqrt(mean(clip(-(rp_gross - cost), 0, inf)^2))`
- 시장/벤치마크 수익 `rm, rb`로 베타를 추정하고, 안정화를 위해 `denom = sign(β)*max(|β|, ε)`
- `Dret = (mu_p_net - mu_b)/denom`, `Treynor = ret_core/denom`
- 최종 보상: `R = w1*ret_core - w2*sigma_down + w3*Dret + w4*Treynor`

---

### 알고리즘(IDQL) 핵심
- **Value(V)**: `expectile` 회귀로 `V`를 목표 `Q_target`에 맞춤
- **Q**: `y = r + γ(1-d)V(s')`를 MSE로 학습
- **정책(확산)**: 상태 임베딩 조건부 VP diffusion에서 epsilon 예측으로 행동 분포를 모사(BC in logits). 폴리시 인코더는 크리틱 인코더의 EMA 복제본으로 안정성 향상
- **행동 선택**: 확산 정책에서 `N`개 후보를 샘플 → 크리틱 `Q`로 점수화 → 최고값 선택(입실론 탐험 시 Dirichlet 샘플)

업데이트 순서(한 스텝):
1) 인코더+V+Q 업데이트(IQL) → 2) `Q_target` 및 폴리시 인코더 EMA 갱신 → 3) 확산 정책 BC 손실 학습

---

### 사용 예시
- 학습/검증은 `src/train.py`에서 자동으로 진행됩니다. 설정에 따라 일정 에피소드마다 검증(`src/environment/validation.py`)이 실행되어, 에이전트 vs 균등분배 vs BTC-온리의 최종 수익률 통계를 남깁니다.
- 체크포인트 활성화 시, 각 검증 시점의 가중치를 저장하고 로드 무결성 점검 후 평가합니다.

추가적인 독립 평가를 원하면:
```python
from src.environment.validation import evaluate_agent
from src.environment.tradingEnv import EnvConfig
from src.data.binance.dataset import DatasetConfig

# data_pre_dict/dataset_cfg/env_cfg는 train.py와 동일하게 구성
summary = evaluate_agent(agent, data_pre_dict, dataset_cfg, env_cfg, n_episodes=8)
print(summary["algo"]["mean_final_return"])  # 평균 최종 수익률
```

---

### 로그/메트릭/체크포인트 확인
- `train.log`: 주요 이벤트/업데이트/검증 기록
- `training_metrics.csv`: 에피소드별 보상/손실/Q/V 평균/리플레이 크기/속도
- `validation_metrics.csv`: 정책별 최종 수익률 통계(평균/중앙/표준편차)
- `models/<run>/episode_XXXXXX_steps_XXXXXXXXX.pt`: 체크포인트(가중치+카운터)

---

### 참고
- Binance Vision: `https://data.binance.vision` (데이터 포맷/주기 등은 변동 가능)


