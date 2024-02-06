from dislib.trees import RandomForestRegressor
import dislib as ds
import numpy as np
from dislib.preprocessing import MinMaxScaler
from dislib.data.array import *
from dislib.data import load_txt_file

def load_event(event_file):
    df = load_txt_file(event_file, discard_first_row=True, col_of_index=True,block_size=(300, 9))
    compss_barrier()
    real_load_data = time.time()
    print("Load real data", real_load_data-ini_time)   
    return df

def load_scalers(x_scaler_file, y_scaler_file):
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.load_model(x_scaler_file)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.load_model(y_scaler_file)
    return scaler_X, scaler_y
      
def load_rf_model(model_file):
    rf = RandomForestRegressor(max_depth=15,n_estimators=15,try_features='third',random_state=0)
    rf.load_model(model, load_format="pickle")
    return rf

def predict(df, rf, scaler_X, scaler_y, result_file):
    x_ds_test = df[:, 0:8]
    x_test = scaler_X.transform(x_ds_test)
    y_pred = scaler_y.inverse_transform(rf.predict(x_test))
    y_pred = y_pred.collect()
    np.savetxt(result_file,y_pred)

if __name__ == '__main__':
    scaler_X,scaler_Y=load_scalers(sys.argv[1], sys.argv[2])
    rf = load_rf_model(sys.argv[3])
    df = load_txt_file(sys.argv[4], discard_first_row=True, col_of_index=True,block_size=(300, 9))
    predict(df, rf, scaler_X, scaler_Y, sys.argv[5])
