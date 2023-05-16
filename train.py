import argparse
import joblib
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0,         help='inverse regularization parameter')
    parser.add_argument('--max_iter', type= int, default=200, help='number of iterations')
    args = parser.parse_args()

    data_url ='https://raw.githubusercontent.com/sierraechobravoindia/udacity-capstone-ml-eng-nd/main/heart_failure_clinical_records_dataset.csv'
    ds = TabularDatasetFactory.from_delimited_files(path=data_url)
    df = ds.to_pandas_dataframe()

    x=df.drop('DEATH_EVENT', axis=1)
    y=df['DEATH_EVENT']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

    run = Run.get_context()
    run.log('regularization strength', np.float(args.C))
    run.log('max iterations', np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy=model.score(x_test, y_test)
    run.log('Accuracy', np.float64(accuracy))
    
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.joblib')

if __name__== '__main__':
    main()



