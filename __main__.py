import spark_job as job

if __name__ == '__main__':
    spark = job.start.initiate_session()
    job.start.run_experiment(spark)