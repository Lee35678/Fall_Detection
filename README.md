

---

# Fall Detection with Camera

📷 **Camera-based Fall Detection using MobileNetV2 + LSTM**

이 프로젝트는 CCTV 또는 웹캠 영상을 기반으로 실시간 낙상(Fall) 여부를 감지하는 딥러닝 파이프라인입니다.
MobileNetV2로 프레임별 특징을 추출하고, LSTM을 이용해 시퀀스 데이터(32프레임)를 분석하여 낙상을 감지합니다.

---

## 🔥 결과

* **Validation Accuracy**: `99.42%`
* **Validation Loss**: `0.0196`

---

## 🚀 주요 파일 구성

| 파일명                         | 설명                                                                   |
| --------------------------- | -------------------------------------------------------------------- |
| `prepare_le2i_auto.py`      | 비디오에서 미리 생성된 `npz` 파일을 불러와 MobileNetV2로 특징 벡터를 추출하여 학습 데이터(.npz)를 생성 |
| `train.py`                  | 추출된 특징 벡터를 이용해 LSTM 모델을 학습하고, `.h5` 파일로 저장                           |
| `run_realtime_detection.py` | 학습된 모델과 MobileNetV2를 이용해 카메라 스트림에서 실시간으로 낙상 여부를 감지                   |

---

## 📂 프로젝트 구조

```
project/
│
├── prepare_le2i_auto.py
├── train.py
├── run_realtime_detection.py
├── le2i_npz/          # 원본 npz 프레임 + 라벨
├── le2i_features/     # MobileNetV2로 추출된 특징 벡터 npz
├── fall_model_light.h5 # 학습된 LSTM 모델
```

---

## ⚙️ 실행 방법

### 1️⃣ 특징 벡터 추출

`prepare_le2i_auto.py`를 실행하여 MobileNetV2로 특징 벡터를 추출합니다.

```bash
python prepare_le2i_auto.py
```

* `/content/drive/MyDrive/Fall_Detector/le2i_npz`에 저장된 프레임 npz를 불러와
* `/content/drive/MyDrive/Fall_Detector/le2i_features`에 특징 벡터 npz를 저장합니다.

### 2️⃣ 모델 학습

`train.py`를 실행하여 LSTM 모델을 학습합니다.

```bash
python train.py
```

* 최적의 모델은 `fall_model_light.h5`로 저장됩니다.

### 3️⃣ 실시간 낙상 감지

`run_realtime_detection.py`를 실행하여 카메라에서 실시간으로 낙상을 감지합니다.

```bash
python run_realtime_detection.py
```

* 화면에 `FALL DETECTED` 라벨이 뜨면 낙상이 감지된 것입니다.
* `q`를 누르면 종료, `s`를 누르면 스크린샷을 저장합니다.

---

## 📈 사용한 데이터셋

본 프로젝트는 [Université Bourgogne Franche-Comté (UBFC)](https://www.ubfc.fr/)에서 제공하는
공식 Fall Detection Dataset ([FR-13002091000019](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)) 을 사용하여 학습하였습니다.

* **Dataset URL:**
  [https://search-data.ubfc.fr/FR-13002091000019-2024-04-09\_Fall-Detection-Dataset.html](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)

* 해당 데이터셋은 실험실 환경에서 수집된 낙상/비낙상 영상 데이터를 포함하고 있으며,
  연구 및 비상업적 목적으로 활용하였습니다.

---

## 🛠 모델 구조

* **Feature Extractor**: MobileNetV2 (`imagenet` pre-trained, `224x224`)
* **Sequence Analyzer**: LSTM(128) → Dropout(0.3) → Dense(1, sigmoid)
* **Input Sequence**: 32 Frames × 1280 features

---

## 💡 특징

✅ MobileNetV2를 통해 연산량을 대폭 줄여 실시간 추론 가능
✅ 가벼운 LSTM 아키텍처로 CPU에서도 동작
✅ 32프레임 시퀀스를 분석해 낙상 이벤트를 robust하게 판별

---

## ✍️ 참고

* `prepare_le2i_auto.py`, `train.py`, `run_realtime_detection.py` 모두 `SEQ_LEN=32`, `HEIGHT=224`, `WIDTH=224` 로 고정되어 있으므로 npz 생성시 동일하게 맞춰주세요.
* OpenCV GUI가 정상 동작하지 않을 경우:

  * 서버 환경에서는 `cv2.imshow` 대신 이미지 저장 방식으로 변경 필요
  * 로컬 환경(Windows/macOS/Linux Desktop)에서 실행 권장

