from imblearn.over_sampling import SMOTE
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

for file in os.listdir('dataset'):
  df = pd.read_csv(f'dataset/{file}')
  categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']

  # Áp dụng LabelEncoder cho mỗi cột phân loại
  label_encoders = {}
  for col in categorical_cols:
      le = LabelEncoder()
      df[col] = le.fit_transform(df[col].astype(str))  # Chuyển sang chuỗi trước khi mã hóa
      label_encoders[col] = le  # Lưu lại le để có thể reverse nếu cần

  # Chia ra các đặc trưng (features) và nhãn (target)
  X = df.drop(columns=['fraud_bool'])
  y = df['fraud_bool']
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Áp dụng SMOTE để cân bằng dữ liệu
  smote = SMOTE(random_state=42)
  X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
  
  df_train_res = pd.concat([X_train_res, y_train_res], axis=1)
  df_train_res.to_csv(f'dataset_resampled/train/{file}', index=False)
  
  df_test = pd.concat([X_test, y_test], axis=1)
  df_test.to_csv(f'dataset_resampled/test/{file}', index=False)
  