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


def set_seed(seed):
    # Python의 랜덤 시드 설정
    random.seed(seed)
    # NumPy의 랜덤 시드 설정
    np.random.seed(seed)
    # PyTorch의 시드 설정
    torch.manual_seed(seed)
    # CUDA를 사용하는 경우에도 동일하게 설정
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 동일한 시드 설정
    # PyTorch의 연산 결과가 재현 가능하도록 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
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


def run_train(use_attention=True, lr=0.01, num_epochs=1, batch_size=32, pred_days=1, ti=None):
    #불러오기
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    
    with open("data/X_train_dates.pkl", "rb") as f:
        X_train_dates = pickle.load(f)
    with open("data/y_train_dates.pkl", "rb") as f:
        y_train_dates = pickle.load(f)


    # Dataset / DataLoader 생성
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 

    '''
    # 첫 번째 배치 확인
    for X_batch, y_batch in train_loader:
        print("X_batch shape:", X_batch.shape)  # (batch_size, 672, 3)
        print("y_batch shape:", y_batch.shape)  # (batch_size, 168, 3)
        break
    '''

    #모델 정의
    set_seed(42)
    params = {'input_dim': 6,
              'hidden_dim': 64,
              'output_dim': pred_days * 24, #7일 * 24시간
              'num_layers': 2}

    if use_attention:
        model = GRUWithAttention(**params)
    else:
        model = GRUModel(**params)

    #손실함수 / 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #모델 학습 / 베스트모델 저장
    best_loss = float('inf')  # 초기 Best Loss 값 설정
    today = datetime.today().strftime('%Y%m%d')
    if use_attention:
        best_model_path = f"gru_with_attention_model_{today}.pth"  #f"train/gru_with_attention_model_{today}.pth" 
    else:
        best_model_path = f"gru_model_{today}.pth" #f"train/gru_model_{today}.pth"


    if mlflow.active_run():
        mlflow.end_run()
    

    mlflow.set_experiment("coin_experiment")

    # 모델 및 메타데이터 저장
    with mlflow.start_run(run_name="Train Model"):
        mlflow.log_param("input_dim", params['input_dim'])
        mlflow.log_param("hidden_dim", params['hidden_dim'])
        mlflow.log_param("output_dim", params['output_dim'])
        mlflow.log_param("num_layers", params['num_layers'])
        
        for epoch in range(num_epochs):
            model.train()
            train_loss=0
            for X_batch, y_batch in tqdm(train_loader):
                optimizer.zero_grad()
                
                # GRU 모델에 입력 (X_batch)
                y_pred = model(X_batch)  # (batch_size, 168, 1)
            
                # # y_batch 차원 맞춤
                # y_batch = y_batch.unsqueeze(-1)  # (batch_size, 168, 1)
            
                # 손실 계산
                loss = criterion(y_pred, y_batch)  # MSELoss 사용
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
        
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
                
            train_loss /= len(train_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch+1)
            
            # # Validation 과정
            # model.eval()
            # val_loss = 0.0
            # with torch.no_grad():
            #     for X_val, y_val in val_loader:  # Validation DataLoader 사용
            #         outputs, _ = model(X_val)
            #         outputs = outputs[:, -1, :]  # 마지막 타임스텝 출력
            #         loss = criterion(outputs, y_val)
            #         val_loss += loss.item()
        
            # val_loss /= len(val_loader)
        
            # Best Model 저장
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch+1
                # torch.save(model.state_dict(), best_model_path)
        
                mlflow.log_metric("best_train_loss", best_loss)
                mlflow.log_metric("best_epoch", best_epoch)
                
                # 모델 저장
                client = MlflowClient()
                try:
                    client.delete_registered_model(name="recent_model")
                except:
                    pass
                mlflow.pytorch.log_model(model, best_model_path, registered_model_name='recent_model') #best_model_path.split('.')[0]
                print(f"Epoch {epoch+1}: Best Model Saved with Train Loss: {train_loss:.4f}")
        
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}") #, Validation Loss: {val_loss:.4f}
        
        print("Training Complete. Best Model Saved at:", best_model_path)



        


if __name__=='__main__':
    run_train()




