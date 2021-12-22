# hello_world.py
from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="data_pipeline1", 
    description="data_pipelinev1", 
    start_date=days_ago(0),
    schedule_interval="*/10 * * * *", # 
    tags=["data"],
) as dag:

    t1 = BashOperator(
        task_id="face_crop",
        bash_command="docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ google_api:0.8 /bin/bash init.sh ",
        owner="yangjae", 
        retries=3, 
        retry_delay=timedelta(minutes=5), 
    )
    t2 = BashOperator(
        task_id="super_resolution",
        bash_command="docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ sure:0.3 /bin/bash /home/myamya/project/init.sh ",
        owner="yangjae", 
        retries=3, 
        retry_delay=timedelta(minutes=5), 
    )
    
    t3 = BashOperator(
        task_id="sketch",
        bash_command="docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ sketch:0.4 /bin/bash /home/myamya/project/init.sh ",
        owner="yangjae", 
        retries=3, 
        retry_delay=timedelta(minutes=5), 
    )
    
    t4 = BashOperator(
        task_id="move_file",
        bash_command="cp /home/myamya/project/image_files/cropped_512_image/* /home/myamya/project/images/real/ | cp /home/myamya/project/image_files/sketched2/* /home/myamya/project/images/sketch/",
        owner="yangjae", 
        retries=3, 
        retry_delay=timedelta(minutes=5), 
    )
    
    t5 = BashOperator(
        task_id="delete_past_images",
        bash_command="rm -rf /home/myamya/project/image_files/images/* | rm -rf /home/myamya/project/image_files/cropped_512_image/* | rm -rf /home/myamya/project/image_files/cropped_image/* | rm -rf /home/myamya/project/image_files/sketched1/* | rm -rf /home/myamya/project/image_files/sketched2/*",
        owner="yangjae", 
        retries=3,
        retry_delay=timedelta(minutes=5), 
    )

t1 >> t2 >> t3 >> t4 >> t5
