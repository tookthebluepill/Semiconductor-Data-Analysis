import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------------------
# 1. 저장된 모델 및 데이터 불러오기
# ----------------------------------------------------------------------
print("--- 1. 모델 및 데이터 불러오기 ---")
model = tf.keras.models.load_model('wafer_model_tuned.keras')
print("모델 로드 완료!")

test_data = np.load('test_data.npz')
X_test = test_data['X_test']
y_test = test_data['y_test']
class_names = np.load('class_names.npy', allow_pickle=True)
print("테스트 데이터 로드 완료!")

# ----------------------------------------------------------------------
# 2. 모델 평가 및 결과 심층 분석
# ----------------------------------------------------------------------
print("\n--- 2. 모델 평가 및 분석 ---")

# 최종 성능 평가
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n테스트 데이터 최종 정확도: {accuracy*100:.2f}%")

# 테스트 데이터에 대한 예측 수행
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 분류 리포트 출력
print("\n[분류 리포트]")
print(classification_report(y_test, y_pred, target_names=class_names))

# 혼동 행렬 생성 및 시각화
print("\n혼동 행렬(Confusion Matrix)을 시각화합니다...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()