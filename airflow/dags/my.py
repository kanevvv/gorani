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
import pendulum

import mlflow
import mlflow.pytorch

from utils.coin_dataloader import run_dataloader_1y #_get_coin_list, _get_data, _connect_DB, _table_to_parquet, 
from utils.preprocessing import run_local_dataload_n_preprocessing_n_save #_data_imputation, _create_sequences,
from utils.Train import run_train #TimeSeriesDataset, set_seed, GRUModel, GRUWithAttention
from utils.Prediction import run_prediction #TimeSeriesDataset, GRUModel, GRUWithAttention
from utils.evaluate import run_evaluate 




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
    dag_id="my",
    start_date= datetime(2025, 1, 22, tzinfo = local_tz), # - timedelta(hours=9), #datetime(2025, 1, 18),
    schedule_interval="@daily",
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
        task_id="run_local_dataload_n_preprocessing_n_save_for_train",
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
        task_id="run_train",
        python_callable=run_train,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_attention" : True,
                   "lr" : LEARNING_RATE,
                   "num_epochs" : NUM_EPOCHS,
                   "batch_size" : 32,
                   "pred_days" : 1
                  },  # 키워드 인자
        provide_context=True,
    )


    
    # PythonOperator 정의
    task4 = PythonOperator(
        task_id="run_prediction_for_train",
        python_callable=run_prediction,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_attention" : USE_ATTENTION,
                   "batch_size" : BATCH_SIZE,
                   "pred_days" : 1,
                   "use_type" : 'train'
                  },  # 키워드 인자 
        provide_context=True,
    )

#############################################
    # PythonOperator 정의
    task5 = PythonOperator(
        task_id="run_local_dataload_n_preprocessing_n_save_for_infer",
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

    
    # PythonOperator 정의
    task6 = PythonOperator(
        task_id="run_prediction_for_infer",
        python_callable=run_prediction,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_attention" : USE_ATTENTION,
                   "batch_size" : BATCH_SIZE,
                   "pred_days" : 1,
                   "use_type" : 'inference'
                  },  # 키워드 인자
        provide_context=True,
    )

#############################################
    # PythonOperator 정의
    task7 = PythonOperator(
        task_id="run_evaluate_for_train",
        python_callable=run_evaluate,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_type" : "train",
                  },  # 키워드 인자
        provide_context=True,
    )
        
    # PythonOperator 정의
    task8 = PythonOperator(
        task_id="run_evaluate_for_infer",
        python_callable=run_evaluate,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_type" : "inference",
                  },  # 키워드 인자
        provide_context=True,
    )


    
    task1 >> task2 >> task3 >> task4 >> task5 >> task6 >> task7 >> task8


