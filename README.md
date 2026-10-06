# Fall Detection with Camera

MobileNetV2로 프레임 특징을 뽑고 LSTM으로 32프레임 시퀀스를 분류해, 웹캠 영상에서 실시간으로 낙상 여부를 판단하는 딥러닝 파이프라인입니다.

## 개요

CCTV 또는 웹캠 영상을 기반으로 낙상(Fall)을 감지하는 것이 목표입니다.
ImageNet으로 사전학습된 MobileNetV2(가중치 고정)로 프레임마다 1280차원 특징 벡터를 추출하고, 32프레임 특징 시퀀스를 가벼운 LSTM 분류기에 넣어 낙상 확률(0~1)을 출력합니다.
특징 추출과 학습은 Google Colab(Google Drive 경로)에서, 실시간 감지는 로컬 PC 웹캠에서 실행하도록 작성되어 있습니다.

## 결과

작성자 학습 로그 기준 수치입니다. `train.py`에서 특징 파일을 무작위로 8:2 분할한 **검증 세트** 결과이며, 별도 테스트 세트 평가는 저장소에 없습니다.

* **Validation Accuracy**: `99.42%`
* **Validation Loss**: `0.0196`

## 주요 기능

* 프레임 npz → MobileNetV2 특징 벡터 npz 일괄 변환 (`prepare_le2i_auto.py`)
* 특징 시퀀스로 LSTM 이진 분류기 학습, `val_accuracy`가 가장 높은 모델을 `.h5`로 저장 (`train.py`)
* 웹캠 실시간 감지: 최근 32프레임 특징을 슬라이딩 윈도우로 유지하며 매 프레임 예측, 확률 > 0.5이면 `FALL DETECTED` 표시 (`run_realtime_detection.py`)
* 사용 가능한 카메라 인덱스(0~9) 자동 탐색, `s` 키로 스크린샷 저장

## 시스템 구성

```
[학습 - Colab]
 le2i_npz/*.npz (frames: 32×224×224×3, label)
   └─ prepare_le2i_auto.py ── MobileNetV2(imagenet, avg pooling) ──> le2i_features/*.npz (features: 32×1280, label)
        └─ train.py ── LSTM(128) → Dropout(0.3) → Dense(1, sigmoid) ──> fall_model_light.h5

[실시간 감지 - 로컬]
 웹캠 프레임 ─ 224×224 리사이즈, /255 정규화 ─ MobileNetV2 ─ 1280 특징
   └─ deque(32) ─ fall_model_light.h5 ─ 확률 > 0.5 ? "FALL DETECTED" : "Normal"
```

| 구성 | 내용 |
|---|---|
| Feature Extractor | MobileNetV2 (`imagenet` pre-trained, `224x224`, `include_top=False`, `pooling='avg'`) |
| Sequence Analyzer | LSTM(128) → Dropout(0.3) → Dense(1, sigmoid) |
| Input Sequence | 32 Frames × 1280 features |
| 학습 설정 | Adam, binary cross-entropy, batch 4, 15 epochs |

## 기술 스택

* Python 3, TensorFlow / Keras
* OpenCV, NumPy
* scikit-learn (`train_test_split`), tqdm
* Google Colab + Google Drive (특징 추출·학습)

## 실행 방법

### 0. 패키지 설치

`requirements.txt`가 없으므로 코드의 import 기준으로 설치합니다.

```bash
pip install tensorflow numpy opencv-python scikit-learn tqdm
```

### 1. 특징 벡터 추출

```bash
python prepare_le2i_auto.py
```

* `/content/drive/MyDrive/Fall_Detector/le2i_npz`에 저장된 프레임 npz(`frames`, `label` 키)를 불러와
* `/content/drive/MyDrive/Fall_Detector/le2i_features`에 특징 벡터 npz(`features`, `label` 키)를 저장합니다.
* 경로는 스크립트 상단 `npz_dir`, `features_dir` 변수에 고정되어 있으므로 Colab 외 환경에서는 수정이 필요합니다.

### 2. 모델 학습

```bash
python train.py
```

* 최적 모델은 `/content/drive/MyDrive/Fall_Detector/fall_model_light.h5`(`model_out_path`)에 저장됩니다.

### 3. 실시간 낙상 감지

```bash
python run_realtime_detection.py
```

* `fall_model_light.h5`를 **현재 작업 폴더**에서 읽습니다 (저장소에 학습된 파일 포함).
* 화면에 `FALL DETECTED` 라벨이 뜨면 낙상이 감지된 것입니다.
* `q`를 누르면 종료, `s`를 누르면 스크린샷을 저장합니다.

## 사용한 데이터셋

본 프로젝트는 [Université Bourgogne Franche-Comté (UBFC)](https://www.ubfc.fr/)에서 제공하는
Fall Detection Dataset ([FR-13002091000019](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html))을 사용하여 학습하였습니다.

* **Dataset URL:**
  [https://search-data.ubfc.fr/FR-13002091000019-2024-04-09\_Fall-Detection-Dataset.html](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)
* 해당 데이터셋은 실험실 환경에서 수집된 낙상/비낙상 영상 데이터를 포함하고 있으며, 연구 및 비상업적 목적으로 활용하였습니다.
* 데이터셋 파일은 저장소에 포함되어 있지 않습니다.

## 폴더 구조

```
Fall_Detection/
├── prepare_le2i_auto.py       # 프레임 npz → MobileNetV2 특징 npz
├── train.py                   # LSTM 학습
├── run_realtime_detection.py  # 웹캠 실시간 감지
└── fall_model_light.h5        # 학습된 LSTM 모델 (약 8.7MB)

# 실행 시 Google Drive에 필요한/생성되는 폴더 (저장소에는 없음)
le2i_npz/        # 원본 프레임 npz + 라벨
le2i_features/   # MobileNetV2 특징 벡터 npz
```

## 참고

* `prepare_le2i_auto.py`, `train.py`, `run_realtime_detection.py` 모두 `SEQ_LEN=32`, `HEIGHT=224`, `WIDTH=224`로 고정되어 있으므로 npz 생성 시 동일하게 맞춰야 합니다. 형태가 다른 파일은 특징 추출 단계에서 건너뜁니다.
* **영상 → 프레임 npz 변환 스크립트는 저장소에 없습니다.** `le2i_npz/`를 만드는 과정(프레임 샘플링, 라벨링 기준)은 코드로 확인할 수 없습니다.
* 검증 분할이 영상 클립 단위 무작위 분할(`random_state=42`)이라, 같은 장소·같은 인물의 클립이 학습/검증에 함께 들어갈 수 있습니다. 위 검증 정확도가 처음 보는 환경에서의 성능을 뜻하지는 않습니다.
* `run_realtime_detection.py`는 `fall_model_light.h5`를 못 읽으면 **학습되지 않은 더미 LSTM 모델**로 대신 실행됩니다. 이때 예측값은 의미가 없으므로 로그의 `[INFO] 테스트용 더미 모델을 생성합니다...` 메시지를 확인하세요.
* 실시간 감지는 매 프레임 MobileNetV2 추론을 하므로 CPU 환경에서는 프레임 속도가 낮을 수 있습니다.
* `prepare_le2i_auto.py`의 "첫 번째 파일 형태 검사"는 `load_npz(all_files)`로 리스트 전체를 넘겨 항상 예외 메시지를 출력합니다(검사만 실패하고 이후 추출은 정상 진행). `all_files[0]`이 의도된 값으로 보입니다.
* OpenCV GUI가 정상 동작하지 않을 경우:
  * 서버 환경에서는 `cv2.imshow` 대신 이미지 저장 방식으로 변경 필요
  * 로컬 환경(Windows/macOS/Linux Desktop)에서 실행 권장
