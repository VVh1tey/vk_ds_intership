import pandas as pd
import numpy as np

from datetime import datetime

import joblib

from src import preprocess

import os
import zipfile

if __name__ == "__main__":
    if os.path.exists(".models.zip"):
        with zipfile.ZipFile('.models.zip', 'r') as zip_ref:
            zip_ref.extractall('./models/')
    
    path = None
    while path is None:
        path = input("Enter local path to data.parquet\nsample path: ./data.parquet\n")
        if os.path.exists(path):
            break

    data = pd.read_parquet(path.strip(), engine='pyarrow')
    
    preprocessed = preprocess.preprocess(data, 'predict')
    ids = data.id   
    del data
    
    model = joblib.load('./models/best_RF_model.pkl')
    
    preds = model.predict_proba(preprocessed)[:, 1]
    
    res = pd.DataFrame({'id': ids, 'preds': preds})

    res.to_csv(f'./output/{(datetime.now()).strftime("%d_%m_%Y_%H_%M")}_submission.csv', index=False)