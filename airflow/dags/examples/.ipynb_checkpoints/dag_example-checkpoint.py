# dags/dag_example.py

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'simple_bash_dag',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval=timedelta(days=1),
)

# 첫 번째 Bash 작업
task1 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag,
)

# 두 번째 Bash 작업
task2 = BashOperator(
    task_id='sleep',
    bash_command='sleep 5',
    dag=dag,
)

# 세 번째 Bash 작업
task3 = BashOperator(
    task_id='print_hello',
    bash_command='echo "Hello World"',
    dag=dag,
)

# 작업 순서 정의
task1 >> task2 >> task3
#출처: https://devops-space.tistory.com/32#DAG (Python) 생성-1 [CW Project:티스토리]