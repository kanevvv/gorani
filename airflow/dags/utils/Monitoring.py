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
# from torch.autograd import Variable
import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset

# from airflow import DAG
# from airflow.operators.python import PythonOperator


import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

def run_monitoring(): #use_type='train', ti=None
    today = datetime.today().date()

    # 둘째날인지 (D-1 inference 파일 존재 여부) 확인
    yesterday = (datetime.today().date() - relativedelta(days=1)).strftime('%Y%m%d')
    two_days_ago = (datetime.today().date() - relativedelta(days=2)).strftime('%Y%m%d')
    if os.path.isfile(f"inference/pred_24h_{yesterday}_use_recent_model.csv") and not os.path.isfile(f"inference/pred_24h_{two_days_ago}_use_recent_model.csv"):
        #실행 둘째날 recnet_model을 best_model로 지정
        client = MlflowClient()
        registered_models = client.search_registered_models()
        registered_model_names = [info.name for info in registered_models]
        if 'best_model' not in registered_model_names:
            for info in registered_models :
                if info.name == 'recent_model':
                    run_id = info.latest_versions[0].run_id
                    break
                    
            
            model_name = 'best_model'  # 모델 이름 생성
            
            try:
                client.delete_registered_model(name=model_name) # 기존 모델 삭제
            except:
                pass
                
            # client = MlflowClient()
            registered_model = client.create_registered_model(model_name)
            artifact_uri = f"runs:/{run_id}/gru_with_attention_model_{yesterday}.pth"
            client.create_model_version(name=model_name, source=artifact_uri, run_id=None)

    try:
        #어제일자 적재 건 불러오기
        yesterday = today - relativedelta(days=1)
        df_true = pd.read_parquet('data/coins.parquet')
        df_true = df_true[df_true['candle_date_time_kst'].dt.date == yesterday][['candle_date_time_kst', 'trade_price']]
        df_true = df_true.sort_values(by='candle_date_time_kst')
        
        #D-1 예측건 불러오기 (Recent model(D-2까지 데이터로 학습), Best model(D-3까지 중))
        yesterday = (today - relativedelta(days=1)).strftime('%Y%m%d')
        df_pred_use_rct_model = pd.read_csv(f'inference/pred_24h_{yesterday}_use_recent_model.csv')
        df_pred_use_best_model = pd.read_csv(f'inference/pred_24h_{yesterday}_use_best_model.csv')
    
        #RMSE 출력 (Recent model, Best model)
        criterion = nn.MSELoss()
        RMSE_rct = torch.sqrt(criterion(torch.tensor(df_true['trade_price'].values), torch.tensor(df_pred_use_rct_model['predicted'].values))).item()
        RMSE_best = torch.sqrt(criterion(torch.tensor(df_true['trade_price'].values), torch.tensor(df_pred_use_best_model['predicted'].values))).item()
        # print("Yesterday Result")
        # print(f"datetime_true : {df_true['candle_date_time_kst'].min()} ~ {df_true['candle_date_time_kst'].max()}")
        # print(f"datetime_pred : {df_pred['tomorrow_datetime'].min()} ~ {df_pred['tomorrow_datetime'].max()}")
        # print(f'RMSE: {RMSE}')
        print(f"RMSE_rct: {RMSE_rct}")
        print(f"RMSE_best: {RMSE_best}")

        
        #mlflow 모델 레지스트리의 best model 갱신
        if RMSE_rct < RMSE_best:
            # if use_attention:
            #     best_model_path = f"gru_with_attention_model_{today}.pth"  #f"train/gru_with_attention_model_{today}.pth" 
            # else:
            #     best_model_path = f"gru_model_{today}.pth" #f"train/gru_model_{today}.pth"
            
            client = MlflowClient()
            registered_models = client.search_registered_models()
            for info in registered_models :
                if info.name == 'recent_model':
                    run_id = info.latest_versions[0].run_id
                    break
                    
            recent_day = (today - relativedelta(days=2)).strftime('%Y%m%d')
            
            model_name = 'best_model'  # 모델 이름 생성
            
            try:
                client.delete_registered_model(name=model_name) # 기존 모델 삭제
            except:
                pass
                
            # client = MlflowClient()
            registered_model = client.create_registered_model(model_name)
            artifact_uri = f"runs:/{run_id}/gru_with_attention_model_{recent_day}.pth"
            client.create_model_version(name=model_name, source=artifact_uri, run_id=None)

            print("best_model is updated as recent_model")
        

    except:
        print("Monitoring can't execute at first day")


if __name__=='__main__':
    run_monitoring()
