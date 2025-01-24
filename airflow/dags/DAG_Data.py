import os
from utils.config import CURRENT_DIR, DB_NAME, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, USE_ATTENTION
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

from airflow import DAG
# from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pendulum

import mlflow
import mlflow.pytorch

from utils.coin_dataloader import run_dataloader_1y #_get_coin_list, _get_data, _connect_DB, _table_to_parquet, 
from utils.preprocessing import run_local_dataload_n_preprocessing_n_save #_data_imputation, _create_sequences,
from utils.Train import run_train #TimeSeriesDataset, set_seed, GRUModel, GRUWithAttention
from utils.Inference import run_inference #TimeSeriesDataset, GRUModel, GRUWithAttention
from utils.Monitoring import run_monitoring





default_args = {
    'owner': 'Harry',
    # 'depends_on_past': False,
    # 'email': ['kwsxfk8332@gmail.com'],
    # 'email_on_failure': False,
    # 'email_on_retry': False,
    'retries': 2, #3
    'retry_delay': timedelta(minutes=1),
}


local_tz = pendulum.timezone("Asia/Seoul")

##############################################################################################################################

# DAG 정의
with DAG(
    dag_id="DAG_Data",
    start_date= datetime(2025, 1, 21, tzinfo = local_tz), # - timedelta(hours=9), #datetime(2025, 1, 18),
    schedule_interval = "00 1 * * *", #"@daily"
    catchup=False,
    default_args=default_args,
) as dag:



    # cli_task1 = BashOperator(
    #     task_id='run_mlflow_ui',
    #     bash_command=f"cd {CURRENT_DIR}; nohup mlflow ui --host 0.0.0.0 --port 5000 & disown",
    #     dag=dag,
    # )


    # PythonOperator 정의
    task1 = PythonOperator(
        task_id="run_dataloader_1y",
        python_callable=run_dataloader_1y,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"coin_name" : "KRW-BTC",
                   "db_name" : DB_NAME,
                   "talbe_name": "coins",
                   "get_coin_list" : False,
                   "table_to_parquet": True,
                  },  # 키워드 인자
    )
        


    # PythonOperator 정의
    task2 = PythonOperator(
        task_id="run_data_preprocessing_for_train",
        python_callable=run_local_dataload_n_preprocessing_n_save,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"coin_name" : "KRW-BTC",
                   "parquet_name" : 'coins.parquet',
                   "use_weeks" : 24,
                   "pred_days" : 1,
                   "use_type" : 'train'
                  },  # 키워드 인자 
        provide_context=True,
    )


    
 
    # PythonOperator 정의
    task3 = PythonOperator(
        task_id="run_data_preprocessing_for_infer",
        python_callable=run_local_dataload_n_preprocessing_n_save,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"coin_name" : "KRW-BTC",
                   "parquet_name" : 'coins.parquet',
                   "use_weeks" : 24,
                   "pred_days" : 1,
                   "use_type" : 'inference'
                  },  # 키워드 인자
        provide_context=True,
    )

###########################################################################################################
    
    # 다음 DAG를 트리거하는 태스크
    trigger_DAG_Train = TriggerDagRunOperator(
        task_id='trigger_DAG_Train',
        trigger_dag_id='DAG_Train',  # 트리거할 DAG의 ID
        wait_for_completion=False,  # DAG 2의 실행이 끝날 때까지 기다리지 않음
    )
    

    # 다음 DAG를 트리거하는 태스크
    trigger_DAG_Monitoring = TriggerDagRunOperator(
        task_id='trigger_DAG_Monitroing',
        trigger_dag_id='DAG_Monitoring',  # 트리거할 DAG의 ID
        wait_for_completion=False,  # DAG 2의 실행이 끝날 때까지 기다리지 않음
    )

    #ㅇㅋ
    task1 >> task2 >> task3 >> [trigger_DAG_Train, trigger_DAG_Monitoring]


