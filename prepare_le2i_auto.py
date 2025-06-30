import tensorflow as tf
import numpy as np
import glob
import os
from tqdm import tqdm

# --- 설정값 ---
# 이 값들이 원본 비디오를 npz로 변환할 때 사용한 값과 동일한지 확인해주세요.
SEQ_LEN = 32
HEIGHT = 224
WIDTH = 224
CH = 3
BATCH_SIZE = 16 # Colab 환경에 맞춰 배치 크기 조절 가능

# --- 경로 설정 ---
# 경로가 정확한지 다시 한번 확인해주세요.
npz_dir = "/content/drive/MyDrive/Fall_Detector/le2i_npz"
features_dir = "/content/drive/MyDrive/Fall_Detector/le2i_features"

# 특징을 저장할 폴더 생성
os.makedirs(features_dir, exist_ok=True)

def build_feature_extractor():
    """특징 추출을 위한 MobileNetV2 모델을 생성합니다."""
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(HEIGHT, WIDTH, CH)
    )
    base.trainable = False
    model = tf.keras.Model(inputs=base.input, outputs=base.output)
    model.compile()
    return model

def load_npz(path):
    """원본 npz 파일에서 프레임과 라벨을 로드합니다."""
    data = np.load(path)
    # npz 파일에 'frames' 키가 있는지 확인
    if 'frames' not in data:
        raise KeyError("npz 파일에 'frames' 키가 없습니다.")
    x = data['frames'].astype("float32") / 255.0
    y = data['label'].astype("float32")
    return x, y

# --- 메인 실행 부분 ---
print("--- 특징 추출 스크립트 시작 ---")

# 1. 경로 및 파일 개수 확인
print(f"입력 데이터 경로: {npz_dir}")
all_files = glob.glob(os.path.join(npz_dir, "*.npz"))
print(f"총 {len(all_files)}개의 npz 파일을 찾았습니다.")

if not all_files:
    print("\n[오류] npz 파일을 찾을 수 없습니다. 'npz_dir' 경로가 올바른지, 구글 드라이브가 마운트되었는지 확인해주세요.")
else:
    # 2. 첫 번째 파일 데이터 형태 확인 (가장 중요한 디버깅 단계)
    print("\n--- 첫 번째 파일 형태 검사 ---")
    try:
        first_x, _ = load_npz(all_files)
        expected_shape = (SEQ_LEN, HEIGHT, WIDTH, CH)
        print(f"첫 번째 파일 '{os.path.basename(all_files)}'의 데이터 형태: {first_x.shape}")
        print(f"스크립트가 기대하는 데이터 형태: {expected_shape}")

        if first_x.shape!= expected_shape:
            print("\n[!!! 중요 경고!!!]")
            print("데이터 형태가 일치하지 않습니다. 이로 인해 모든 파일이 처리되지 않았을 가능성이 높습니다.")
            print("스크립트 상단의 SEQ_LEN, HEIGHT, WIDTH, CH 값을 확인하고,")
            print("이전에 비디오를 npz로 변환할 때 사용했던 값과 동일하게 맞춰주세요.")
        else:
            print("데이터 형태가 올바릅니다. 계속 진행합니다.")

    except Exception as e:
        print(f"\n[오류] 첫 번째 파일을 로드하는 중 문제가 발생했습니다: {e}")
        print("npz 파일이 손상되었거나 내부 구조가 다를 수 있습니다.")

    # 3. 특징 추출기 생성 및 실행
    print("\n--- 특징 추출 시작 ---")
    feature_extractor = build_feature_extractor()

    skipped_count = 0
    processed_count = 0

    for file_path in tqdm(all_files, desc="특징 추출 진행률"):
        try:
            frames, label = load_npz(file_path)

            if frames.shape!= (SEQ_LEN, HEIGHT, WIDTH, CH):
                skipped_count += 1
                continue

            features = feature_extractor.predict(frames, batch_size=BATCH_SIZE, verbose=0)

            base_filename = os.path.basename(file_path)
            output_path = os.path.join(features_dir, base_filename)

            np.savez(output_path, features=features, label=label)
            processed_count += 1

        except Exception as e:
            print(f"\n파일 {os.path.basename(file_path)} 처리 중 오류 발생: {e}")
            skipped_count += 1

    print("\n--- 특징 추출 완료 ---")
    print(f"총 {len(all_files)}개 파일 중:")
    print(f"  - 처리 성공: {processed_count}개")
    print(f"  - 건너뜀: {skipped_count}개")
    print(f"\n추출된 특징은 다음 경로에 저장되었습니다: {features_dir}")