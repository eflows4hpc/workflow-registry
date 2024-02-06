import dislib as ds
import numpy as np
import sys
import pandas as pd
import json
from dislib.regression import RandomForestRegressor
from dislib.model_selection import GridSearchCV
from dislib.preprocessing import MinMaxScaler
from dislib.utils import train_test_split
from dislib.data import load_txt_file

def write_results(results_path, cv_results, best_model):
    pd_df = pd.DataFrame.from_dict(cv_results)
    pd_df.to_csv(results_path +'/pd.csv')
    best_model.save_model(results_path + '/model.dat', save_format='pickle')    

def model_selection(input_dataset, results_path, parameters):
    df = load_txt_file(input_dataset ,discard_first_row=True, col_of_index=True,block_size=(133334, 9))
    Data_X_arr = df[:, 0:8]
    Data_Y_arr = df[:, 8:9]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(Data_X_arr)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(Data_Y_arr)
    x_ds_train, x_ds_test, y_ds_train, y_ds_test = train_test_split(Data_X_arr, Data_Y_arr)
    x_ds_test = x_ds_test.rechunk((100000, 8))
    y_ds_test = y_ds_test.rechunk((100000, 1))
    x_train = scaler_X.transform(x_ds_train)
    x_test = scaler_X.transform(x_ds_test)
    y_train = scaler_y.transform(y_ds_train)
    y_test = scaler_y.transform(y_ds_test)
    rf = RandomForestRegressor(n_estimators=15, try_features = 'third')
    searcher = GridSearchCV(rf, parameters, cv=3)
    np.random.seed(0)
    searcher.fit(x_train, y_train)
    write_results(results_path, searcher.cv_results_, searcher.best_estimator_) 

if __name__ == "__main__":
   dataset_file = sys.argv[1]
   results_path = sys.argv[2]
   f = open(sys.argv[3])
   parameters = json.load(f)
   model_selection(dataset_file, results_path, parameters)
