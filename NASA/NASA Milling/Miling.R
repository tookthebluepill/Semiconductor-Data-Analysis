# 1. 패키지 설치 (아직 설치하지 않았다면)
install.packages("R.matlab")

# 2. 패키지 불러오기
library(R.matlab)

# 3. .mat 파일 읽기
mat_data <- readMat("C:/Users/Christina/Documents/R_workspace1-main/NASA/NASA Milling/mill/mill.mat")

# 4. 불러온 데이터 구조 확인
# list 안에 어떤 변수들이 있는지 이름을 보여줍니다.
names(mat_data)

# list의 전체적인 구조를 자세히 봅니다.
str(mat_data)

# 최종 결과를 담을 빈 리스트를 만듭니다.
results_list <- list()


# 5. 데이터 사용하기
# 167번의 실험을 하나씩 순회합니다.
for (i in 1:167) {
  
  
  # i번째 실험 데이터에 접근합니다.
  experiment_data <- mat_data$mill[,,i]
  
  
  # i번째 실험의 조건(metadata)을 추출합니다.
  # VB는 tool wear, 즉 공구 마모도로 가장 중요한 타겟 변수입니다.
  
  case_id <- experiment_data[[1]]
  tool_wear <- experiment_data[[3]]
  
  
  
  # i번째 실험의 6개 시계열 센서 데이터를 추출합니다.
  
  # dimnames를 참고하면 8~13번째가 센서 데이터입니다.
  smcAC <- experiment_data[[8]]
  smcDC <- experiment_data[[9]]
  vib_spindle <- experiment_data[[10]]
  vib_table <- experiment_data[[11]]
  ae_spindle <- experiment_data[[12]]
  ae_table <- experiment_data[[13]]
  
  # i번째 실험의 실제 데이터 길이를 가져옵니다. (nrow는 행의 개수를 세어줍니다)
  actual_length <- nrow(smcAC)
  
  # i번째 실험 데이터를 하나의 데이터 프레임으로 만듭니다.
  
  temp_df <- data.frame(
    case = case_id,
    VB = tool_wear,
    time_step = 1:actual_length, # 9000개의 시간 순서
    smcAC = smcAC,
    smcDC = smcDC,
    vib_spindle = vib_spindle,
    vib_table = vib_table,
    ae_spindle = ae_spindle,
    ae_table = ae_table
    
  )
  
  
  
  # 완성된 i번째 데이터 프레임을 리스트에 추가합니다.
  results_list[[i]] <- temp_df
  
}



# 리스트에 담긴 167개의 데이터 프레임을 하나의 큰 데이터 프레임으로 합칩니다.
# dplyr 패키지가 필요할 수 있습니다. install.packages("dplyr")
final_df <- dplyr::bind_rows(results_list)





# 최종 결과 확인
cat("최종 데이터 프레임의 차원(행, 열):", dim(final_df), "\n")
head(final_df) # 앞부분 6줄 출력
tail(final_df) # 뒷부분 6줄 출력



##탐색적 데이터 분석(EDA)
#공구 마모도(VB)에 따라 센서 값의 분포가 어떻게 달라지는지 확인하기 위해 박스 플롯으로 확인
#install.packages("ggplot2")
library(ggplot2)

# VB 값을 그룹(factor)으로 만들어 박스 플롯 생성
# VB(마모도)가 커질수록 smcAC 센서 값의 분포가 어떻게 변하는지 확인
ggplot(final_df, aes(x = factor(VB), y = smcAC)) +
  geom_boxplot() +
  labs(
    title = "공구 마모도(VB)에 따른 smcAC 센서 값 분포",
    x = "공구 마모도 (VB)",
    y = "smcAC 센서 값"
  ) +
  theme_minimal()

# dplyr 패키지가 로드되어 있는지 확인
library(dplyr)
library(ggplot2)


## 데이터 정제(Data Cleaning)
# 1. 데이터 정제
#    - VB가 NaN이 아닌 데이터만 선택
#    - smcAC의 값이 비정상적으로 크거나 작은 이상치를 제거
cleaned_df <- final_df %>%
  filter(!is.na(VB)) %>%
  filter(abs(smcAC) < 1e10) # 1e10 보다 큰 극단치를 제거 (필요시 이 숫자를 조정)


# 2. 정제된 데이터로 다시 시각화
ggplot(cleaned_df, aes(x = factor(VB), y = smcAC)) +
  geom_boxplot() +
  labs(
    title = "공구 마모도(VB)에 따른 smcAC 센서 값 분포 (정제 후)",
    x = "공구 마모도 (VB)",
    y = "smcAC 센서 값"
  ) +
  theme_minimal() +
  # X축 라벨이 너무 많아 겹치므로, 일부만 표시되도록 수정
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


## IQR을 이용한 이상치 제거
# dplyr, ggplot2 패키지 로드
library(dplyr)
library(ggplot2)

# 1. IQR 계산을 위한 통계량 산출
Q1 <- quantile(final_df$smcAC, 0.25, na.rm = TRUE)
Q3 <- quantile(final_df$smcAC, 0.75, na.rm = TRUE)
IQR_value <- IQR(final_df$smcAC, na.rm = TRUE) # 또는 Q3 - Q1

# 2. 정상 데이터 범위 설정
lower_bound <- Q1 - 1.5 * IQR_value
upper_bound <- Q3 + 1.5 * IQR_value

# 3. 데이터 정제 (NaN 제거 및 IQR 기반 이상치 제거)
cleaned_df_iqr <- final_df %>%
  filter(!is.na(VB)) %>%
  filter(smcAC >= lower_bound, smcAC <= upper_bound)


# 4. 최종 정제된 데이터로 시각화
ggplot(cleaned_df_iqr, aes(x = factor(VB), y = smcAC)) +
  geom_boxplot() +
  labs(
    title = "공구 마모도(VB)에 따른 smcAC 센서 값 분포 (IQR 정제 후)",
    x = "공구 마모도 (VB)",
    y = "smcAC 센서 값"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


## 특징 공학(Feature Engineering)
library(dplyr)

# case와 VB를 기준으로 그룹화하고, 각 센서의 표준편차(sd)를 계산
features_df <- cleaned_df_iqr %>%
  group_by(case, VB) %>%
  summarise(
    smcAC_sd = sd(smcAC),
    smcDC_sd = sd(smcDC),
    vib_spindle_sd = sd(vib_spindle),
    # ... 다른 센서들의 표준편차도 추가 ...
    .groups = 'drop'
  )

# 결과 확인
head(features_df)


## 머신러닝으로 마모도 예측하기(랜덤 포레스트)
# 1. randomForest 패키지 설치 및 로드
# install.packages("randomForest")
library(randomForest)

# 2. 모델 학습
# VB를 예측하기 위해 _sd로 끝나는 모든 변수를 사용합니다.
# (실제로는 데이터를 학습용/테스트용으로 나누지만, 여기서는 개념 설명을 위해 전체 데이터로 학습)
model <- randomForest(VB ~ ., data = features_df %>% select(VB, ends_with("_sd")))


# 3. 모델 학습 결과 확인
# 모델이 VB의 분산을 약 몇 % 설명하는지(예측하는지) 보여줍니다.
print(model)


# 4. 변수 중요도 확인 (가장 중요한 단계)
# 어떤 센서의 변동성(sd)이 마모도 예측에 가장 중요한지 보여줍니다.
importance(model)
varImpPlot(model)



## 머신러닝으로 마모도 예측하기(xgboost)
# 1. xgboost 패키지 설치 및 로드
#install.packages("xgboost")
library(xgboost)
library(dplyr)

# 2. XGBoost용 데이터 준비
# (데이터 프레임을 숫자 행렬(matrix) 형태로 변환해야 합니다)

# 예측 변수(features)와 타겟 변수(label) 분리
features_data <- features_df %>%
  select(ends_with("_sd")) %>% # _sd로 끝나는 열(센서 변동성) 선택
  as.matrix()

label_data <- features_df$VB # 타겟 변수(마모도)

# XGBoost 전용 데이터 형식인 DMatrix로 변환
dtrain <- xgb.DMatrix(data = features_data, label = label_data)


# 3. XGBoost 모델 학습
# verbose = 0 옵션은 학습 과정을 화면에 출력하지 않아 간결합니다.
xgb_model <- xgboost(
  data = dtrain,
  nrounds = 100, # 부스팅 라운드(트리의 개수)를 100으로 설정
  objective = "reg:squarederror", # 회귀(Regression) 문제 설정
  verbose = 0
)


# 4. 모델 학습 결과 확인
print(xgb_model) # 모델의 상세 정보 확인


# 5. 변수 중요도 확인 및 시각화
importance_matrix <- xgb.importance(model = xgb_model)

cat("--- XGBoost 변수 중요도 ---\n")
print(importance_matrix)

xgb.plot.importance(importance_matrix)