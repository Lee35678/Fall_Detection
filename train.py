import tensorflow as tf
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

# 상수 설정
SEQ_LEN = 32
FEATURE_DIM = 1280  # MobileNetV2의 출력 벡터 크기
BATCH = 4
EPOCHS = 15

# --- 데이터 처리 함수들 ---

def load_features_npz(path):
    """사전 추출된 특징 벡터와 라벨을 로드합니다."""
    data = np.load(path)
    x = data['features'].astype("float32")
    y = data['label'].astype("float32")
    return x, y

def generator(file_list):
    """특징 벡터 파일 리스트로부터 데이터를 무한히 생성하는 제너레이터입니다."""
    local_files = file_list.copy()
    while True:
        # 매 에포크 시작 시 데이터를 섞어줍니다.
        np.random.shuffle(local_files)
        for f in local_files:
            x, y = load_features_npz(f)
            yield x, y

def make_dataset(files):
    """특징 벡터를 위한 tf.data.Dataset을 생성합니다."""
    ds = tf.data.Dataset.from_generator(
        lambda: generator(files),
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LEN, FEATURE_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_light_model():
    """
    MobileNetV2가 제거된 매우 가벼운 LSTM 모델을 생성합니다.
    """
    inputs = tf.keras.Input(shape=(SEQ_LEN, FEATURE_DIM))
    x = tf.keras.layers.LSTM(128, return_sequences=False)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 메인 실행 부분 ---

# 특징 벡터가 저장된 폴더와 모델을 저장할 경로
features_dir = "/content/drive/MyDrive/Fall_Detector/le2i_features"
model_out_path = "/content/drive/MyDrive/Fall_Detector/fall_model_light.h5"

# 모든 특징 벡터 파일(.npz)을 찾습니다.
all_files = glob.glob(os.path.join(features_dir, "*.npz"))

if not all_files:
    print(f"오류: '{features_dir}'에서 특징 파일을 찾을 수 없습니다.")
    print("먼저 extract_features.py 스크립트를 실행하여 특징을 추출해주세요.")
else:
    print(f"총 {len(all_files)}개의 특징 벡터 파일을 찾았습니다.")

    # 학습용/검증용 데이터 분리
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # TF Dataset 생성
    train_ds = make_dataset(train_files)
    val_ds = make_dataset(val_files)

    # 경량화된 모델 생성 및 구조 확인
    model = build_light_model()
    model.summary()

    # 에포크당 스텝 수 계산
    steps_per_epoch = len(train_files) // BATCH
    validation_steps = len(val_files) // BATCH

    # ModelCheckpoint 콜백 정의
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_out_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # 모델 학습 시작
    print("\n경량화된 모델로 학습을 시작합니다...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        # 수정된 부분: 콜백을 리스트 형태로 올바르게 전달합니다.
        callbacks=[checkpoint_callback]
    )
    print("\n학습이 완료되었습니다.")