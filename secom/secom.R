# 1. 필요 라이브러리 설치 및 로딩
install.packages(c("tidyverse", "tidymodels", "themis", "ranger", "xgboost", "bonsai", "lightgbm", "vip"))

library(tidyverse)
library(tidymodels)
library(themis)
library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)
library(vip)

# ----------------------------------------------------------------
# 2. 데이터 준비 및 전처리 레시피 (이전과 동일)
# ----------------------------------------------------------------
secom_df <- read_csv("secom_combined.csv") %>%
  mutate(label = as.factor(ifelse(label == -1, "Pass", "Fail")))

set.seed(42)
data_split <- initial_split(secom_df, prop = 0.8, strata = label)
train_data <- training(data_split)
test_data  <- testing(data_split)

secom_recipe <- recipe(label ~ ., data = train_data) %>%
  update_role(timestamp, new_role = "ID") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_smote(label, over_ratio = 1)

# ----------------------------------------------------------------
# 3. 여러 모델 사양을 리스트로 정의
# ----------------------------------------------------------------
rf_spec <- rand_forest(trees = 200) %>%
  set_engine("ranger") %>%
  set_mode("classification")

lgbm_spec <- boost_tree(trees = 200, learn_rate = 0.05) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

xgb_spec <- boost_tree(trees = 200, learn_rate = 0.05) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 모델들을 명확한 이름과 함께 리스트로 묶습니다.
model_list <- list(
  random_forest = rf_spec,
  lightgbm = lgbm_spec,
  xgboost = xgb_spec
)

# ----------------------------------------------------------------
# 4. Workflow Set 생성 및 모든 모델 실행
# ----------------------------------------------------------------
# 교차 검증(cross-validation)을 위한 데이터 폴드(fold)를 만듭니다.
set.seed(42)
cv_folds <- vfold_cv(train_data, v = 5, strata = label)

# 워크플로우 셋 생성
# preproc 에는 레시피 리스트를, models 에는 모델 리스트를 넣습니다.
secom_wflow_set <- workflow_set(
  preproc = list(recipe = secom_recipe),
  models = model_list
)

# 모든 워크플로우를 교차 검증 데이터에 대해 실행합니다. (시간이 다소 소요됩니다)
# 평가지표로 f1-score를 지정합니다.
f_measure <- metric_set(f_meas)

secom_results <- workflow_map(
  secom_wflow_set,
  resamples = cv_folds,
  fn = "fit_resamples", # fit_resamples 함수를 모든 워크플로우에 적용
  metrics = f_measure,
  verbose = TRUE
)

# ----------------------------------------------------------------
# 5. 최종 결과 비교
# ----------------------------------------------------------------
print("### 모든 모델 최종 성능 비교 ###")

# 성능이 좋은 순서대로 순위를 매겨 보여줍니다.
rank_results(secom_results)

# 결과를 그래프로 시각화
autoplot(secom_results)