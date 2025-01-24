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
    dag_id="DAG_Inference",
    start_date= datetime(2025, 1, 21, tzinfo = local_tz), # - timedelta(hours=9), #datetime(2025, 1, 18),
    schedule_interval = None, #"30 0 * * *", #"@daily"
    catchup=False,
    default_args=default_args,
) as dag:


    # PythonOperator 정의
    task1 = PythonOperator(
        task_id="run_inference_using_recent_model",
        python_callable=run_inference,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_attention" : USE_ATTENTION,
                   "batch_size" : BATCH_SIZE,
                   "pred_days" : 1,
                   "use_model_type" : 'recent_model'
                  },  # 키워드 인자 
        provide_context=True,
    )
    


    # PythonOperator 정의
    task2 = PythonOperator(
        task_id="run_inference_using_best_model",
        python_callable=run_inference,  # 호출할 함수
        # op_args=["value1", "value2"],  # 위치 기반 인자
        op_kwargs={"use_attention" : USE_ATTENTION,
                   "batch_size" : BATCH_SIZE,
                   "pred_days" : 1,
                   "use_model_type" : 'best_model'
                  },  # 키워드 인자 
        provide_context=True,
    )


    



    #ㅇㅋ
    task1 >> task2



