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


def run_evaluate(use_type='train', ti=None):
    today = datetime.today().date()

    if use_type=='train':
        true_days=2
        pred_days=1

    if use_type=='inference':
        true_days=1
        pred_days=1


    try:
        #어제일자 적재 건 불러오기
        yesterday = today - relativedelta(days=true_days)
        df_true = pd.read_parquet('data/coins.parquet')
        df_true = df_true[df_true['candle_date_time_kst'].dt.date == yesterday][['candle_date_time_kst', 'trade_price']]
        df_true = df_true.sort_values(by='candle_date_time_kst')
        
        #모델(이틀전까지 데이터로 학습)의 어제 예측건 불러오기
        yesterday = (today - relativedelta(days=pred_days)).strftime('%Y%m%d')
        df_pred = pd.read_csv(f'inference/pred_24h_{use_type}_{yesterday}.csv')
    
        #RMSE 출력
        criterion = nn.MSELoss()
        RMSE=torch.sqrt(criterion(torch.tensor(df_true['trade_price'].values), torch.tensor(df_pred['predicted'].values))).item()
        print("Yesterday Result")
        print(f"datetime_true : {df_true['candle_date_time_kst'].min()} ~ {df_true['candle_date_time_kst'].max()}")
        print(f"datetime_pred : {df_pred['tomorrow_datetime'].min()} ~ {df_pred['tomorrow_datetime'].max()}")
        print(f'RMSE: {RMSE}')


        #Xcom에서 run_id / experiment_id 가져오기  
        run_id = ti.xcom_pull(key='run_id')
        experiment_id = ti.xcom_pull(key='experiment_id')

        mlflow.set_experiment(experiment_id=experiment_id)
        
        # 모델 및 메타데이터 저장
        with mlflow.start_run(run_id=run_id):
            # 모델 관련 정보 저장
            mlflow.log_param(f"{use_type}_datetime_true", f"{df_true['candle_date_time_kst'].min()} ~ {df_true['candle_date_time_kst'].max()}")
            mlflow.log_param(f"{use_type}_datetime_pred", f"{df_pred['tomorrow_datetime'].min()} ~ {df_pred['tomorrow_datetime'].max()}")
            mlflow.log_metric("RMSE", RMSE)

    except:
        pass


if __name__=='__main__':
    run_evaluate()
