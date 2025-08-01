# ----------------------------------------------------------------------
# 0. 환경 설정 및 라이브러리 임포트
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation

# ----------------------------------------------------------------------
# 1. 데이터 로드 및 클리닝
# ----------------------------------------------------------------------
print("--- 1. 데이터 로드 및 클리닝 시작 ---")

def clean_label(label):
    if isinstance(label, list) and len(label) > 0 and isinstance(label[0], list) and len(label[0]) > 0:
        return label[0][0]
    if isinstance(label, np.ndarray) and label.size > 0:
        return label.item(0)
    return label

df = pd.read_pickle("LSWMD.pkl")
df['failureType'] = df['failureType'].apply(clean_label)
df['trianTestLabel'] = df['trianTestLabel'].apply(clean_label)
print("데이터 로드 및 클리닝 완료!")

# ----------------------------------------------------------------------
# 2. 데이터 전처리 및 분리
# ----------------------------------------------------------------------
print("\n--- 2. 데이터 전처리 및 분리 시작 ---")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("데이터 전처리 및 분리 완료!")

# ----------------------------------------------------------------------
# 3. CNN 모델 구축 (용량 증대)
# ----------------------------------------------------------------------
print("\n--- 3. CNN 모델 구축 및 학습 시작 (용량 증대) ---")

data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    RandomRotation(0.2),
], name="data_augmentation")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 모델 용량 증대: Conv2D 레이어 추가 및 Dense 뉴런 수 증가
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
num_classes = len(faulty_types)
model = Sequential([
    data_augmentation,
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Conv2D 레이어 한 층 추가
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    # Dense 레이어의 뉴런 수를 128 -> 256으로 증가
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
], name="wafer_classifier_large")


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ----------------------------------------------------------------------
# 4. 모델 학습
# ----------------------------------------------------------------------
print("\n모델 학습을 시작합니다 (용량 증대된 모델)...")
model.fit(X_train, y_train,
          epochs=30,
          batch_size=64,
          validation_data=(X_test, y_test))

# ----------------------------------------------------------------------
# 5. 학습된 모델과 데이터 저장
# ----------------------------------------------------------------------
print("\n--- 5. 모델 및 데이터 저장 ---")

model.save('wafer_model_large.keras')
print("용량 증대된 모델을 'wafer_model_large.keras' 파일로 저장했습니다.")

np.savez('test_data.npz', X_test=X_test, y_test=y_test)
np.save('class_names.npy', faulty_types)
print("분석용 데이터를 'test_data.npz', 'class_names.npy' 파일로 저장했습니다.")