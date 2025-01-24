from utils.config import CURRENT_DIR
import os
os.chdir(CURRENT_DIR)

import requests
from requests.exceptions import RequestException
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import pickle
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# from airflow import DAG
# from airflow.operators.python import PythonOperator


import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1])  # 마지막 타임스텝의 hidden state 사용
        return out


class GRUWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUWithAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = nn.Linear(hidden_dim, 1)  # Attention 스코어 계산
        
    def forward(self, x):
        gru_out, _ = self.gru(x)  # GRU 출력 (batch_size, seq_len, hidden_dim)
        
        # Attention 스코어 계산
        attention_scores = self.attention(gru_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        # Context Vector 계산 (가중합)
        context = torch.sum(gru_out * attention_weights, dim=1)  # (batch_size, hidden_dim)

        # 출력 계산
        out = self.fc(context)  # (batch_size, output_dim)
        return out #, attention_weights  # Attention 가중치도 반환 (시각화 가능)



def run_inference(use_attention=True, batch_size=32, pred_days=1, use_model_type='best_model'): 
    client = MlflowClient()
    registered_models = client.search_registered_models()


    #불러오기
    X_test = np.load("data/X_inference.npy")
    
    with open("data/X_inference_dates.pkl", "rb") as f:
        X_test_dates = pickle.load(f)



    
    # Dataset / DataLoader 생성
    test_dataset = TimeSeriesDataset(X_test, X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    '''  
    # 첫 번째 배치 확인
    for X_batch, y_batch in test_loader:
        print("X_batch shape:", X_batch.shape)  # (batch_size, 672, 3)
        print("y_batch shape:", y_batch.shape)  # (batch_size, 168, 3)
        break
    '''




    #모델 불러오기 ( from mlfow repository)
    if use_model_type in ['best_model', 'recent_model']:
        try:
            model =  mlflow.pytorch.load_model(model_uri=f"models:/{use_model_type}/latest")
        except:
            print('best_model is empty!')
            return 

    else:
        print( f"check use_model_type={use_model_type}" )
        return
    
    # 모델 평가 모드 설정
    model.eval()
    
    # 예측 결과를 저장할 리스트
    all_predictions = []
    
    # DataLoader를 사용해 예측
    with torch.no_grad():  # 그래디언트 비활성화
        for X_batch, _ in tqdm(test_loader): 
            # 모델을 사용해 예측
            y_pred = model(X_batch)  # [batch_size, 168, 1]
            # y_pred = y_pred.squeeze(-1)  # [batch_size, 168]
            all_predictions.append(y_pred)
    
    # 전체 결과 병합
    all_predictions = torch.cat(all_predictions, dim=0)  # [슬라이딩 윈도우개수=샘플개수, 168]

    # # Y Scaler 불러오기
    # with open("data/mm_y.pkl", "rb") as f:
    #     mm_y = pickle.load(f)

    # MLflow에서 Y Scaler 로드
    mm_y_path = f"data/mm_y.pkl"
    with open(mm_y_path, "rb") as f:
        mm_y = pickle.load(f)  

    # 예측값과 실제값을 원래 스케일로 되돌림
    # 텐서를 넘파이 배열로 변환
    i=-1 #마지막 시퀀스 기준 예측
    predicted = all_predictions[i].detach().numpy()  # 기울기 추적 비활성화
    predicted = mm_y.inverse_transform(predicted.reshape(1,-1))


    # #RMSE 출력
    # criterion = nn.MSELoss()
    # RMSE=torch.sqrt(criterion(torch.tensor(predicted), torch.tensor(label_y))).item()
    # print('Train data : three days ago')
    # print("Test data : two days ago")
    # print(f'RMSE: {RMSE}')


    next_first_time = (X_test_dates[-1].tail(1) + timedelta(hours=1)).values[0] #데이터의 가장 최근 일자
    tomorrow_datetime = pd.date_range(start=next_first_time, periods=24, freq='h')

    today = datetime.today().date().strftime('%Y%m%d')
    # yesterday = (today - relativedelta(days=1)).strftime('%Y%m%d')
    
    pred_24h = pd.DataFrame({'tomorrow_datetime' : tomorrow_datetime, 'predicted': predicted[0]})
    pred_24h.to_csv(f'inference/pred_24h_{today}_use_{use_model_type}.csv', index=False)




if __name__=='__main__':
    run_inference(use_model_type='best_model')
    run_inference(use_model_type='recent_model')


