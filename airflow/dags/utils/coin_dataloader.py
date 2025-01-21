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


import mlflow
import mlflow.pytorch


def _get_coin_list():
    url = "https://api.upbit.com/v1/market/all?is_details=true"
    headers = {"accept": "application/json"}
    res = requests.get(url, headers=headers)
    coin_list = [[r['market'],r['korean_name'],r['english_name']]  for r in res.json()]
    
    df_coin_list = pd.DataFrame(coin_list, columns = ['market', 'korean_name','english_name'])
    df_coin_list = df_coin_list[df_coin_list['market'].str.startswith('KRW')]
    df_coin_list.reset_index(inplace=True, drop=True)
    # len = len(self.df_coin_list)
    df_coin_list.to_csv('data/df_coin_list.csv', index=False)
    # return df_coin_list    


def _get_data(market='KRW-BTC', minute=60, cnt=200, to=''):
    '''
    cnt<=200
    '''
    # {market}에 2024년 10월 1일(UTC) 이전 가장 최근 {minute}분봉 {cnt}개를 요청
    url = f"https://api.upbit.com/v1/candles/minutes/{minute}"
    
    params = {'market': market,  
              'count': cnt,
              'to': to #미입력시 지금 기준, "2024-10-01 00:00:00"
             }

    headers = {"accept": "application/json"}
    
    response = requests.get(url, params=params, headers=headers)

    return response.json()


def _connect_DB(data_json, db_name='test_db', table_name='coins', drop_table=False) -> None:
    '''
    DB연동
    '''
    # 데이터베이스 연결
    conn = sqlite3.connect(db_name)
    
    # 커서 생성
    cursor = conn.cursor()
    
    #테이블 드랍
    if drop_table:
        cursor.execute('drop table if exists coins')
    
    # 테이블 생성
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            market TEXT, --PRIMARY KEY,
            candle_date_time_utc TEXT,
            candle_date_time_kst TEXT,
            opening_price FLOAT,
            high_price FLOAT,
            low_price FLOAT,
            trade_price FLOAT,
            timestamp TEXT,
            candle_acc_trade_price FLOAT,
            candle_acc_trade_volume FLOAT,
            unit INTEGER
        )
    ''')

    LEN = len(data_json) #len(response.json())
    for idx in range(LEN):
        columns = list(data_json[idx].keys())
        values = list(data_json[idx].values())
        values[:3] = list(map(lambda x: '"' + x + '"', values[:3]))
        values = list(map(str, values))
        
        columns = ', '.join(columns)
        values = ', '.join(values)
        
        # 데이터 삽입
        cursor.execute(f'INSERT INTO coins ({columns}) VALUES ({values})')

    # 트랜잭션 커밋
    conn.commit()
    
    
    # 연결 종료
    conn.close()

    # df = pd.DataFrame(data=rows, columns=columns)
    # return df    




def _table_to_parquet(table_name = 'coins', db_name='test.db', parquet_name='coins.parquet') -> None:
    
    # 데이터베이스 연결
    conn = sqlite3.connect(os.path.join('data', db_name))
    
    # 커서 생성
    cursor = conn.cursor()
    
    #실행 및 결과 가져오기
    cursor.execute(f'SELECT * FROM {table_name}')
    rows = cursor.fetchall()
    # for row in rows:
    #     print(row)
    
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    columns = list(map(lambda x: x[1], columns_info))
    
    
    # 연결 종료
    conn.close()
    
    #df 생성
    df = pd.DataFrame(data=rows, columns=columns)
    # df.drop_duplicates(inplace=True)
    
    #datetime 변환
    date_columns = ['candle_date_time_utc',
                    'candle_date_time_kst']
    for date_col in date_columns:
        df[f'{date_col}'] = pd.to_datetime(df[f'{date_col}'])

    df.to_parquet(os.path.join('data', parquet_name))



def run_dataloader_1y(coin_name='KRW-BTC', db_name='test.db', table_name='coins', get_coint_list=False, table_to_parquet=True) -> None:
    if get_coint_list:
        _get_coin_list()
    '''
    약 1년치 (366일) 적재
    - api 호출 시 데이터 200개 제한
    '''
    yesterday_23h = datetime.today().replace(hour=0, minute=0,second=0, microsecond=0) #- relativedelta(days=1)
    yesterday_23h += relativedelta(days=-1, hours=15) #kst to utc 변환 (utc 기준 시차 반영)
    bf_1_year = yesterday_23h - relativedelta(years=1)

    
    times = []
    i=0
    while True:
        date = yesterday_23h - relativedelta(hours=200*i) #200시간 전
        if date <= bf_1_year:
            break
        times.append(date.strftime("%Y-%m-%d %H:%M:%S"))
    
        i+=1


    
    '''
    coin_name 설정 시, 해당 코인정보만 DB에 적재
    coin_name 미설정 시, 전체 코인정보 DB에 적재
    '''
    if coin_name:
        kw_coin_list = [coin_name]
    else:
        df_coin_list = pd.read_csv('data/df_coin_list.csv')
        kw_coin_list = list(df_coin_list['market'])

    for kw_coin in tqdm(kw_coin_list):
        for i in range(len(times)):
                '''
                - times에 있는 timestamp만 try
                - 한 timestamp 당 최대 5번 try 후, 안되면 timestamp의 iteration으로 continue
                '''
                drop_table=False
                if i==0:
                    drop_table=True
                    
                try:
                    for attempt in range(5):
                        try:
                            data_json = _get_data(market=kw_coin, minute=60, cnt=200, to=times[i])
                            _connect_DB(data_json, db_name=os.path.join('data', db_name), table_name=table_name, drop_table=drop_table)
                            break
                        except RequestException as e:
                            print(f"Attempt {attempt + 1} failed: {e}")
                            time.sleep(2)  # 2초 대기후 재시도
                except:
                    continue


    if table_to_parquet:
        _table_to_parquet(table_name=table_name, db_name=db_name, parquet_name='coins.parquet')


if __name__=='__main__':
    run_dataloader_1y()
