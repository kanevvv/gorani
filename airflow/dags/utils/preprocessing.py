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

# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset

# from airflow import DAG
# from airflow.operators.python import PythonOperator

from airflow.models import TaskInstance

import mlflow
import mlflow.pytorch

def _data_imputation(df : pd.DataFrame):
    #시간순 정렬
    df.sort_values(by='candle_date_time_kst', inplace=True)

    #시간기준 데이터 보간
    start_time = df['candle_date_time_kst'].min()
    end_time = df['candle_date_time_kst'].max()
    full_time = pd.DataFrame( pd.date_range(start=start_time, end=end_time, freq='h'), columns=['full_time'])
    df_full = pd.merge(full_time, df, how='left', left_on='full_time', right_on='candle_date_time_kst')
    df_full.drop(['candle_date_time_kst'], axis=1, inplace=True)
    df_full = df_full.interpolate(method='linear')
    df_full.columns = ['candle_date_time_kst', 'market', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume']
    
    return df_full   


# 슬라이딩 윈도우 생성 함수
def _create_sequences(data, datetimes, input_length, output_length, use_type='train'):
    if use_type=='train':
        X, y = [], []
        X_dates, y_dates = [], []
        for i in tqdm(range(len(data) - input_length - output_length + 1)):
            X.append(data[i : i + input_length])  # 입력: 672시간
            y.append(data[i + input_length : i + input_length + output_length,3])  # 출력: 168시간, 'trade_price' 컬럼
    
            X_dates.append(datetimes[i : i + input_length])
            y_dates.append(datetimes[i + input_length : i + input_length + output_length]) 

        return np.array(X), np.array(y), X_dates, y_dates

    if use_type=='inference':
        X = []
        X_dates = []
        for i in tqdm(range(len(data) - input_length + 1)):
            X.append(data[i : i + input_length])  # 입력: 672시간
            X_dates.append(datetimes[i : i + input_length])

    
        return np.array(X), [] , X_dates, []



def run_local_dataload_n_preprocessing_n_save(coin_name='KRW-BTC', parquet_name='coins.parquet', use_weeks=24, pred_days=1, use_type='train', ti=None):
    '''
    
    '''
    #데이터로드 & 결측치대체
    df = pd.read_parquet(os.path.join('data',parquet_name))
    tmp = df[df['market']==coin_name]
    tmp = tmp.drop(['candle_date_time_utc', 'unit', 'timestamp'], axis=1)
    tmp.sort_values(by='candle_date_time_kst', inplace=True)
    tmp.reset_index(inplace=True, drop=True)
    tmp = _data_imputation(tmp)

    if use_type=='train':
        #Train/Test 분할
        ##train : 가장 최근일자 ~ 24주전(6개월 전)
        thr2 = tmp['candle_date_time_kst'].max() #train_end_time  #시작일 + 8주, #인자 옵션 : train_end_time = datetime(2024,10,1,0), test_end_time = datetime(2024,12,1,0)
        thr1 = thr2 - timedelta(weeks= use_weeks)  #시작일
        Train = tmp[(tmp['candle_date_time_kst'] >= thr1) & (tmp['candle_date_time_kst'] <= thr2)].iloc[:, 2:]
        Train_dates = tmp[(tmp['candle_date_time_kst'] >= thr1) & (tmp['candle_date_time_kst'] <= thr2)]['candle_date_time_kst']

        ##test :  train 과 동일하게 가져가면 미래 예측
        thr4 = tmp['candle_date_time_kst'].max() #thr1 - timedelta(hours=1) #test_end_time  #시작일 + 8주
        thr3 = thr4 - timedelta(weeks= use_weeks) #thr2 + timedelta(days=1) #시작일
        Test = tmp[(tmp['candle_date_time_kst'] >= thr3) & (tmp['candle_date_time_kst'] <= thr4)].iloc[:, 2:]
        Test_dates = tmp[(tmp['candle_date_time_kst'] >= thr3) & (tmp['candle_date_time_kst'] <= thr4)]['candle_date_time_kst']

        #스케일링
        mm_X = MinMaxScaler() # 0 ~ 1
        mm_y = MinMaxScaler()

        X_train_scaled = mm_X.fit_transform(Train)
        y_train_scaled = mm_y.fit_transform(Train[['trade_price']]) #역변환시 사용
        
        X_test_scaled = mm_X.transform(Test)
        y_test_scaled = mm_y.transform(Test[['trade_price']]) #역변환시 사용

        #슬라이딩 윈도우 생성
        '''
        - X_train, y_train, X_train_dates, y_train_dates / X_test, y_test, X_test_dates, y_test_dates 생성
        - (use_weeks//2)W * 7D * 24h (2016시간) 입력, #28일(672시간) 입력
        - 7일(168시간) 출력
         X_train shape: (506, 672, 6) = (윈도우개수, 입력시간, Feature 개수)
         y_train shape: (506, 168) = (윈도우개수, 출력시간, Feature 개수)
         X_test shape: (506, 672, 6)
         y_test shape: (506, 168)
        '''
        input_length = (use_weeks//2) * 7 * 24 #슬라이딩 윈도우해야 하므로 use_weeks의 절반만 사용
        output_length = pred_days * 24 #7일 * 24시간

        X_train, y_train, X_train_dates, y_train_dates = _create_sequences(X_train_scaled, Train_dates, input_length, output_length, use_type='train')
        X_test, y_test, X_test_dates, y_test_dates = _create_sequences(X_test_scaled, Test_dates, input_length, output_length, use_type='train')
        
        #저장
        np.save("data/X_train.npy", X_train)
        np.save("data/y_train.npy", y_train)
        np.save("data/X_test.npy", X_test)
        np.save("data/y_test.npy", y_test)

        
        with open("data/X_train_dates.pkl", "wb") as f:
            pickle.dump(X_train_dates, f)
        with open("data/y_train_dates.pkl", "wb") as f:
            pickle.dump(y_train_dates, f)
        with open("data/X_test_dates.pkl", "wb") as f:
            pickle.dump(X_test_dates, f)
        with open("data/y_test_dates.pkl", "wb") as f:
            pickle.dump(y_test_dates, f) 

        
        #M scaler 저장
        with open("data/mm_X.pkl", "wb") as f:
            pickle.dump(mm_X, f)
        with open("data/mm_y.pkl", "wb") as f:
            pickle.dump(mm_y, f)
    

        
            
    if use_type=='inference':
        #Train/Test 분할
        ##train : 가장 최근일자 ~ 24주전(6개월 전)
        thr2 = tmp['candle_date_time_kst'].max() #train_end_time  #시작일 + 8주, #인자 옵션 : train_end_time = datetime(2024,10,1,0), test_end_time = datetime(2024,12,1,0)
        thr1 = thr2 - timedelta(weeks= use_weeks)  #시작일
        Train = tmp[(tmp['candle_date_time_kst'] >= thr1) & (tmp['candle_date_time_kst'] <= thr2)].iloc[:, 2:]
        Train_dates = tmp[(tmp['candle_date_time_kst'] >= thr1) & (tmp['candle_date_time_kst'] <= thr2)]['candle_date_time_kst']

        ##test :  train 과 동일하게 가져가면 미래 예측
        thr4 = tmp['candle_date_time_kst'].max() #thr1 - timedelta(hours=1) #test_end_time  #시작일 + 8주
        thr3 = thr4 - timedelta(weeks= use_weeks) #thr2 + timedelta(days=1) #시작일
        Test = tmp[(tmp['candle_date_time_kst'] >= thr3) & (tmp['candle_date_time_kst'] <= thr4)].iloc[:, 2:]
        Test_dates = tmp[(tmp['candle_date_time_kst'] >= thr3) & (tmp['candle_date_time_kst'] <= thr4)]['candle_date_time_kst']

        
        # MLflow에서 Scaler 로드
        mm_X_path = f"data/mm_X.pkl"
        with open(mm_X_path, "rb") as f:
            mm_X = pickle.load(f)


        X_test_scaled = mm_X.transform(Test)
        

        input_length = (use_weeks//2) * 7 * 24 #슬라이딩 윈도우해야 하므로 use_weeks의 절반만 사용
        output_length = pred_days * 24 #7일 * 24시간

        X_test, _, X_test_dates, _ = _create_sequences(X_test_scaled, Test_dates, input_length, output_length, use_type='inference')

        
        #저장
        np.save("data/X_inference.npy", X_test)
        
        with open("data/X_inference_dates.pkl", "wb") as f:
            pickle.dump(X_test_dates, f)




if __name__=='__main__':
    run_local_dataload_n_preprocessing_n_save(use_type='train')
    run_local_dataload_n_preprocessing_n_save(use_type='inference')
    