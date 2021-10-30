from spark_job import start

if __name__ == '__main__':
    spark = start.initiate_session()
    start.run_experiment(spark)