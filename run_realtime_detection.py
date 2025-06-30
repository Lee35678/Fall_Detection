# 파일명: run_realtime_detection_fixed.py
# VSCode용 낙상감지 코드 (수정버전)

import tensorflow as tf
import numpy as np
import cv2
from collections import deque
import sys
import os

# --- 상수 설정 ---
SEQ_LEN = 32
FEATURE_DIM = 1280
HEIGHT = 224
WIDTH = 224
CH = 3

# --- 모델 및 특징 추출기 로드 ---
print("[INFO] 모델을 로드하는 중입니다...")
try:
    # 학습된 LSTM 모델 로드
    model = tf.keras.models.load_model("fall_model_light.h5")
    print("[INFO] 모델 로드 성공!")
except Exception as e:
    print(f"[오류] 모델 파일(fall_model_light.h5)을 로드할 수 없습니다: {e}")
    print("현재 디렉토리:", os.getcwd())
    print("디렉토리 내 파일들:", os.listdir("."))
    
    # 더미 모델 생성 (테스트용)
    print("[INFO] 테스트용 더미 모델을 생성합니다...")
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(SEQ_LEN, FEATURE_DIM)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("[INFO] 더미 모델 생성 완료!")

def build_feature_extractor():
    """
    실시간 프레임에서 특징을 추출하기 위한 MobileNetV2 모델을 생성합니다.
    """
    print("[INFO] 특징 추출기(MobileNetV2)를 로드합니다...")
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(HEIGHT, WIDTH, CH)
    )
    base.trainable = False
    feature_extractor = tf.keras.Model(inputs=base.input, outputs=base.output)
    feature_extractor.compile()
    print("[INFO] 특징 추출기 로드 완료!")
    return feature_extractor

# 특징 추출기 생성
feature_extractor = build_feature_extractor()

# --- 카메라 초기화 함수 ---
def initialize_camera():
    """여러 카메라 인덱스를 시도하여 사용 가능한 카메라를 찾습니다."""
    print("[INFO] 사용 가능한 카메라를 찾는 중...")
    
    # 여러 카메라 인덱스 시도
    for camera_index in range(10):  # 0부터 9까지 시도
        print(f"[INFO] 카메라 인덱스 {camera_index}를 시도합니다...")
        cap = cv2.VideoCapture(camera_index)
        
        # 카메라가 열렸는지 확인
        if cap.isOpened():
            # 실제로 프레임을 읽을 수 있는지 테스트
            ret, frame = cap.read()
            if ret:
                print(f"[성공] 카메라 인덱스 {camera_index}에서 카메라를 찾았습니다!")
                print(f"[INFO] 프레임 크기: {frame.shape}")
                return cap
            else:
                print(f"[실패] 카메라 인덱스 {camera_index}: 프레임을 읽을 수 없습니다.")
                cap.release()
        else:
            print(f"[실패] 카메라 인덱스 {camera_index}: 카메라를 열 수 없습니다.")
            cap.release()
    
    print("[오류] 사용 가능한 카메라를 찾을 수 없습니다.")
    return None

def safe_window_operations():
    """OpenCV 창 작업을 안전하게 처리합니다."""
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[경고] 창 닫기 오류 (무시 가능): {e}")

# --- 실시간 감지 준비 ---
sequence = deque(maxlen=SEQ_LEN)

# 웹캠 초기화
cap = initialize_camera()

if cap is None:
    print("\n" + "="*50)
    print("카메라를 사용할 수 없어 프로그램을 종료합니다.")
    print("해결 방법:")
    print("1. 다른 프로그램에서 카메라를 사용 중인지 확인")
    print("2. 카메라 드라이버가 제대로 설치되어 있는지 확인")
    print("3. 카메라 권한이 허용되어 있는지 확인")
    print("4. USB 카메라라면 연결 상태 확인")
    print("="*50)
    sys.exit(1)

print("[INFO] 실시간 낙상 감지를 시작합니다...")
print("[INFO] 종료하려면 화면에서 'q' 키를 누르거나 터미널에서 Ctrl+C를 누르세요")

# 카메라 설정 최적화
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

# --- 실시간 감지 루프 ---
frame_count = 0
try:
    while True:
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break

        frame_count += 1
        
        # 화면 표시용 프레임 복사
        display_frame = frame.copy()

        # --- 프레임 전처리 및 특징 추출 ---
        # 1. 모델 입력에 맞게 프레임 크기 조정 (224x224)
        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
        # 2. 픽셀 값 정규화 (0-1 사이)
        normalized_frame = resized_frame.astype("float32") / 255.0
        # 3. 배치 차원 추가
        input_frame = np.expand_dims(normalized_frame, axis=0)
        
        try:
            # 4. MobileNetV2를 통해 특징 벡터 추출
            features_with_batch_dim = feature_extractor.predict(input_frame, verbose=0)
            
            # (1, 1280) 형태를 (1280,) 형태로 변경하여 불필요한 차원 제거
            squeezed_features = np.squeeze(features_with_batch_dim)
            
            # 5. (1280,) 형태의 순수 벡터를 시퀀스에 추가
            sequence.append(squeezed_features)

            # --- 예측 및 결과 표시 ---
            # 시퀀스가 32개 프레임으로 채워졌을 때만 예측 수행
            if len(sequence) == SEQ_LEN:
                # (32, 1280) 형태의 배열에 배치 차원을 추가하여 (1, 32, 1280) 형태로 변환
                input_sequence = np.expand_dims(np.array(sequence), axis=0)
                
                # 모델 예측
                prediction = model.predict(input_sequence, verbose=0)
                
                # ★★★ 수정된 부분: 배열에서 스칼라 값을 제대로 추출 ★★★
                pred_value = float(prediction[0][0])  # 2차원 배열에서 스칼라 값 추출
                
                # 예측 결과에 따라 라벨과 색상 결정
                if pred_value > 0.5:
                    label = "FALL DETECTED"
                    color = (0, 0, 255)  # 빨간색
                    print(f"[경고] 낙상 감지! 확률: {pred_value:.3f}")
                else:
                    label = "Normal"
                    color = (0, 255, 0)  # 초록색
                    
                # 화면에 텍스트 표시
                text = f"{label}: {pred_value:.3f}"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 프레임 카운터 표시
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # 시퀀스가 아직 채워지지 않았을 때
                progress_text = f"Collecting: {len(sequence)}/{SEQ_LEN}"
                cv2.putText(display_frame, progress_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            print(f"[오류] 특징 추출 또는 예측 중 오류: {e}")
            error_text = "Processing Error"
            cv2.putText(display_frame, error_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 결과 프레임 화면에 표시
        try:
            cv2.imshow("Real-time Fall Detection", display_frame)
        except Exception as e:
            print(f"[오류] 화면 표시 오류: {e}")
            print("OpenCV GUI 지원에 문제가 있을 수 있습니다.")
            break

        # 'q' 키를 누르면 루프 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 사용자가 'q' 키를 눌러 종료합니다.")
            break
        elif key == ord('s'):
            # 's' 키로 스크린샷 저장
            screenshot_name = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_name, display_frame)
            print(f"[INFO] 스크린샷 저장: {screenshot_name}")

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C로 프로그램을 중단합니다.")
except Exception as e:
    print(f"[오류] 예상치 못한 오류가 발생했습니다: {e}")
finally:
    # --- 자원 해제 ---
    print("[INFO] 자원을 해제하고 프로그램을 종료합니다.")
    if cap is not None:
        cap.release()
    safe_window_operations()

print("[INFO] 프로그램이 정상적으로 종료되었습니다.")