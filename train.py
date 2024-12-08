import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

def train_lightgbm(dataset):
  df = pd.read_csv(f'dataset_resampled/train/{dataset}')
  X = df.drop(columns=['fraud_bool'])
  y = df['fraud_bool']
  
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  # Tạo bộ dữ liệu LightGBM
  train_data_res = lgb.Dataset(X_train, label=y_train)
  test_data = lgb.Dataset(X_val, label=y_val, reference=train_data_res)

  # Cấu hình tham số LightGBM
  params = {
      'objective': 'binary',
      'metric': 'binary_error',
      'boosting_type': 'gbdt',
      'num_leaves': 31,
      'learning_rate': 0.05,
      'feature_fraction': 0.9
  }
  num_round = 100

  # Huấn luyện mô hình với LightGBM
  bst = lgb.train(params, train_data_res, num_round, valid_sets=[test_data])
  dataset_name = dataset.split('.')[0]
  bst.save_model(f'lightgbm_{dataset_name}.txt')
  
def train_xgboost(dataset):
  df = pd.read_csv(f'dataset_resampled/train/{dataset}')
  X = df.drop(columns=['fraud_bool']) 
  y = df['fraud_bool']
  
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dval = xgb.DMatrix(X_val, label=y_val)
  
  params = {
      'objective': 'binary:logistic',  # Phân loại nhị phân
      'eval_metric': 'auc',            # Đánh giá AUC
      'n_estimators': 1000,            # Số lượng cây quyết định
      'random_state': 42,
      'learning_rate': 0.031,           # Giảm tốc độ học
      'max_depth': 8,                  # Độyy sâu tối đa của mỗi cây
      'min_child_weight': 100,           # Tối thiểu trọng số của mỗi đứa trẻ (giúp giảm overfitting và cải thiện recall)
      'subsample': 0.9,                # Tỷ lệ mẫu được lấy ngẫu nhiên trong mỗi cây
      'colsample_bytree': 0.8,         # Tỷ lệ các đặc trưng ngẫu nhiên được chọn cho mỗi cây
      'gamma': 1,                      # Tham số điều chỉnh độ phức tạp của mô hình
      'scale_pos_weight': 1,           # Tăng trọng số lớp dương để giúp mô hình học tốt hơn với lớp gian lận
      'lambda': 1,                     # L2 regularization
      'alpha': 0.5,                    # L1 regularization
      'early_stopping_rounds': 200,     # Dừng huấn luyện nếu không cải thiện được trong 50 vòng lặp
  }
  
  bst = xgb.train(params, dtrain, 200, evals=[(dval, 'eval')])
  dataset_name = dataset.split('.')[0]
  bst.save_model(f'xgboost_{dataset_name}.json')
  
datasets = os.listdir('dataset_resampled/train')
for dataset in datasets:
  train_lightgbm(dataset)
  train_xgboost(dataset)
