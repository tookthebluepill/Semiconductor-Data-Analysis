# ----------------------------------------------------------------------
# 0. 환경 설정 및 라이브러리 임포트
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg') # VS Code 환경에서 그래프 창을 띄우기 위한 설정

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. 데이터 로드 및 클리닝
# ----------------------------------------------------------------------
print("--- 1. 데이터 로드 및 클리닝 시작 ---")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 수정된 부분: 강력한 클리닝 함수 정의
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def clean_label(label):
    """
    다양한 형태(이중 리스트, numpy 배열)의 라벨을 순수 문자열로 변환합니다.
    """
    # Case 1: [['Center']] 같은 이중 리스트 형태
    if isinstance(label, list) and len(label) > 0 and isinstance(label[0], list) and len(label[0]) > 0:
        return label[0][0]
    # Case 2: array(['Center']) 같은 numpy 배열 형태
    if isinstance(label, np.ndarray) and label.size > 0:
        return label.item(0)
    # Case 3: 이미 'Center' 처럼 깨끗한 문자열 형태
    return label
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# 데이터 로드
df = pd.read_pickle("LSWMD.pkl")

# 수정된 클리닝 함수 적용
df['failureType'] = df['failureType'].apply(clean_label)
df['trianTestLabel'] = df['trianTestLabel'].apply(clean_label)

# waferMap의 크기를 저장하는 'waferMapDim' 컬럼 추가
df['waferMapDim'] = df.waferMap.apply(lambda x: x.shape)

print(f"데이터 로드 및 클리닝 완료! (전체 데이터 수: {len(df)})")
print("클리닝 후 데이터 샘플:")
print(df.head())

# (이하 코드는 이전과 동일합니다)

# ----------------------------------------------------------------------
# 2. 데이터 탐색 (EDA)
# ----------------------------------------------------------------------
print("\n--- 2. 데이터 탐색(EDA) 시작 ---")

df_faulty = df[df['failureType'] != 'none'].copy()
print(f"정상(none) 데이터 제외 후, 불량 데이터 수: {len(df_faulty)}")

plt.figure(figsize=(10, 6))
df_faulty['failureType'].value_counts().plot(kind='bar', color='salmon')
plt.title('Distribution of Faulty Wafer Types', fontsize=16)
plt.xlabel('Failure Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

print("\nWafer Map 크기(Dimension) 분포:")
print(df_faulty['waferMapDim'].value_counts().head())

faulty_types = df_faulty['failureType'].unique()
num_types = len(faulty_types)
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle('Sample Wafer Maps for Each Failure Type', fontsize=16)
axes = axes.ravel()

for i, f_type in enumerate(faulty_types):
    if i < len(axes):
        sample_row = df_faulty[df_faulty['failureType'] == f_type].iloc[0]
        wafer_map = sample_row['waferMap']
        
        ax = axes[i]
        ax.imshow(wafer_map)
        ax.set_title(f"Type: {f_type}\nDim: {wafer_map.shape}")
        ax.set_xticks([])
        ax.set_yticks([])

for i in range(num_types, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
print("\n데이터 탐색(EDA) 그래프를 화면에 표시합니다...")
plt.show()

# ----------------------------------------------------------------------
# 3. 모델링을 위한 데이터 전처리
# ----------------------------------------------------------------------
print("\n--- 3. 모델링을 위한 데이터 전처리 시작 ---")

IMG_SIZE = 32
print(f"모든 Wafer Map을 ({IMG_SIZE}, {IMG_SIZE}) 크기로 리사이징합니다.")

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

faulty_types_map = {label: i for i, label in enumerate(faulty_types)}
df_faulty['failureCode'] = df_faulty['failureType'].map(faulty_types_map)

print("데이터 리사이징 및 라벨 인코딩 완료!")
print("인코딩된 라벨 정보:")
print(faulty_types_map)

# ----------------------------------------------------------------------
# 4. 학습/테스트 데이터셋 분리
# ----------------------------------------------------------------------
print("\n--- 4. 학습/테스트 데이터셋 분리 시작 ---")

X = np.array(df_faulty['waferMap_resized'].tolist())
y = np.array(df_faulty['failureCode'].tolist())
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("데이터셋 분리 완료!")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n\n모든 전처리 과정이 완료되었습니다. 이제 이 데이터로 딥러닝 모델을 학습시킬 수 있습니다.")

# ----------------------------------------------------------------------
# 5. CNN 모델 구축 및 학습 (새로 추가되는 부분)
# ----------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("\n--- 5. CNN 모델 구축 및 학습 시작 ---")

# 모델 정의
num_classes = len(faulty_types) # 불량 유형의 개수

model = Sequential([
    # 입력 이미지 형태: (32, 32, 1)
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(), # 2D 이미지를 1D 벡터로 변환
    
    Dense(128, activation='relu'),
    Dropout(0.5), # 과적합 방지를 위한 Dropout
    
    # 출력층: 클래스 개수만큼의 뉴런과 softmax 활성화 함수 사용
    Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # 다중 분류 문제, y가 정수일 때 사용
              metrics=['accuracy'])

# 모델 구조 요약 출력
model.summary()

# 모델 학습
print("\n 모델 학습을 시작합니다...")
history = model.fit(X_train, y_train,
                    epochs=20, # 전체 데이터를 20번 반복 학습
                    batch_size=64,
                    validation_data=(X_test, y_test))

# ----------------------------------------------------------------------
# 6. 모델 평가 및 결과 시각화
# ----------------------------------------------------------------------
print("\n--- 6. 모델 평가 및 결과 시각화 ---")

# 테스트 데이터로 최종 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n 테스트 데이터 최종 정확도: {accuracy*100:.2f}%")

# 학습 과정 시각화 (정확도 및 손실)
plt.figure(figsize=(12, 5))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --- 7. 모델 결과 심층 분석 ---
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("\n--- 7. 모델 결과 심층 분석 ---")

# 테스트 데이터에 대한 예측 수행
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 클래스 이름 리스트 생성
class_names = list(faulty_types_map.keys())

# 분류 리포트 출력
print("\n[분류 리포트]")
print(classification_report(y_test, y_pred, target_names=class_names))

# 혼동 행렬 생성 및 시각화
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()