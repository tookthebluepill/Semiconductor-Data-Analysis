# ----------------------------------------------------------------------
# 0. 환경 설정 및 라이브러리 임포트
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation

# ----------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 (train.py와 동일)
# ----------------------------------------------------------------------
print("--- 1. 데이터 로드 및 전처리 시작 ---")

def clean_label(label):
    if isinstance(label, list) and len(label) > 0 and isinstance(label[0], list) and len(label[0]) > 0:
        return label[0][0]
    if isinstance(label, np.ndarray) and label.size > 0:
        return label.item(0)
    return label

df = pd.read_pickle("LSWMD.pkl")
df['failureType'] = df['failureType'].apply(clean_label)
df['trianTestLabel'] = df['trianTestLabel'].apply(clean_label)

df['waferMapDim'] = df.waferMap.apply(lambda x: x.shape)
df_faulty = df[df['failureType'] != 'none'].copy()
IMG_SIZE = 32

def resize_wafer(wafer_map, size=IMG_SIZE):
    h, w = wafer_map.shape
    zoom_factor = size / max(h, w)
    resized_map = zoom(wafer_map, zoom_factor)
    h_new, w_new = resized_map.shape
    pad_h = (size - h_new) / 2
    pad_w = (size - w_new) / 2
    pad_top, pad_bottom = int(np.floor(pad_h)), int(np.ceil(pad_h))
    pad_left, pad_right = int(np.floor(pad_w)), int(np.ceil(pad_w))
    return np.pad(resized_map, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

df_faulty['waferMap_resized'] = df_faulty.waferMap.apply(resize_wafer)
faulty_types = df_faulty['failureType'].unique()
faulty_types_map = {label: i for i, label in enumerate(faulty_types)}
df_faulty['failureCode'] = df_faulty['failureType'].map(faulty_types_map)

X = np.array(df_faulty['waferMap_resized'].tolist())
y = np.array(df_faulty['failureCode'].tolist())
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("데이터 전처리 및 분리 완료!")


# ----------------------------------------------------------------------
# 2. 튜닝을 위한 모델 빌드 함수 정의
# ----------------------------------------------------------------------
def build_model(hp):
    """KerasTuner가 호출할 모델 생성 함수"""
    model = Sequential()
    
    # 데이터 증강 레이어
    model.add(RandomFlip("horizontal_and_vertical", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(RandomRotation(0.2))
    
    # Conv2D 레이어 1
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=64, step=32),
        kernel_size=(3,3),
        activation='relu'
    ))
    model.add(MaxPooling2D((2, 2)))
    
    # Conv2D 레이어 2
    model.add(Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=32),
        kernel_size=(3,3),
        activation='relu'
    ))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    # Dense 레이어
    model.add(Dense(
        units=hp.Int('dense_units', min_value=128, max_value=256, step=128),
        activation='relu'
    ))
    
    # Dropout 레이어
    model.add(Dropout(
        rate=hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
    ))
    
    # 출력 레이어
    model.add(Dense(len(faulty_types), activation='softmax'))
    
    # 학습률(Learning Rate) 튜닝
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# ----------------------------------------------------------------------
# 3. KerasTuner 설정 및 탐색 실행
# ----------------------------------------------------------------------
print("\n--- 3. 하이퍼파라미터 튜닝 시작 ---")

# Hyperband 튜너 설정
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy', # 최적화 목표: 검증 데이터 정확도
                     max_epochs=20,            # 각 모델을 최대 몇 epoch까지 학습할지
                     factor=3,
                     directory='keras_tuner_dir', # 튜닝 결과를 저장할 폴더
                     project_name='wafer_fault_tuning')

# 탐색 실행 (이 과정이 매우 오래 걸릴 수 있습니다)
# 조기 종료 콜백: 검증 손실이 5번 연속 개선되지 않으면 학습 중단
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[stop_early])

# ----------------------------------------------------------------------
# 4. 최적 모델 저장
# ----------------------------------------------------------------------
print("\n--- 4. 최적 모델 및 하이퍼파라미터 결과 ---")

# 최적의 하이퍼파라미터들 가져오기
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
최적의 하이퍼파라미터 조합을 찾았습니다:
- conv_1_filter: {best_hps.get('conv_1_filter')}
- conv_2_filter: {best_hps.get('conv_2_filter')}
- dense_units: {best_hps.get('dense_units')}
- dropout_rate: {best_hps.get('dropout_rate')}
- learning_rate: {best_hps.get('learning_rate')}
""")

# 최적의 하이퍼파라미터로 모델을 다시 빌드하고 전체 데이터로 재학습
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 재학습된 최적 모델 저장
model.save('wafer_model_tuned.keras')
print("\n튜닝된 최적 모델을 'wafer_model_tuned.keras' 파일로 저장했습니다.")