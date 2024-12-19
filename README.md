# Streamlit application for Bank Fraud Detection
## CS313 - Data Mining and Application - Group 2
- Tran Kim Ngoc Ngan (22520002)
- Tran Nhu Cam Nguyen (22520004)
- Duong Pham Hoang Anh (22520042)
## Set up
- Clone this repo
  ```
  git clone https://github.com/cnmeow/bank-fraud-detection
  cd bank-fraud-detection
  ```
- Install all dependencies
  ```
  pip install -r requirement.txt
  ```
- Run Streamlit application
  ```
  streamlit run Home.py
  ```
- Then, application will be run at [http://localhost:8501](http://localhost:8501)

## Dataset
- We use data from the [Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), which includes 6 data files (`Base.csv`, `Variant I.csv`, `Variant II.csv`, `Variant III.csv`, `Variant IV.csv`, `Variant V.csv`)
- The original data is stored in the `dataset` folder.
- The data after applying SMOTE for balancing is stored in the `dataset_resampled` folder, where we have divided it into two subfolders `train` and `test`.
- You can select a dataset to view the data plotting in the `Home` tab. The left column shows the plotting before applying SMOTE, while the right column shows the plotting after applying SMOTE
  ![image](https://github.com/user-attachments/assets/04587294-76b6-47aa-9b76-dfe1e7b2ac92)
## Fraud Detection
### Select a dataset for demo
- You can add your own data file to the `data_resampled/test folder`, as long as it follows the same template as the other data files.
- Alternatively, you can edit the content of the existing data files in the `data_resampled/test` folder. Or simply use the existing data files.
### Let's see the results
- The left column shows the results of XGBoost, while the right column displays the results of LightGBM on the dataset you selected.
- We have pre-trained two models on the six available data files. You just need to select the appropriate checkpoint to get the corresponding results.

![image](https://github.com/user-attachments/assets/649bc154-b451-45aa-88fa-6056a64b0e08)
