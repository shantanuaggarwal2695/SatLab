
if __name__ == '__main__':
    import spark_job.start as start
    spark = start.initiate_session()
    start.run_experiment(spark)