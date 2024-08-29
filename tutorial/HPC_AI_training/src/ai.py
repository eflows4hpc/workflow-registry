from dislib.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import pickle
import importlib
import dislib as ds
from sklearn.metrics import r2_score
import os

def model_selection(Y, X, model_parameters, results_dir):
    X = ds.array(X, block_size=X.shape)
    Y = Y[:, np.newaxis]
    Y = ds.array(Y, block_size=Y.shape)
    model = load_model(model_parameters.get("model"))
    params = model_parameters.get("parameters")
    searcher = GridSearchCV(model, params, cv=4, scoring=r2_score)
    searcher.fit(X, Y)
    save_results(searcher, results_dir)

def load_model(model):
    last_dot_index = model.rfind('.')
    if last_dot_index != -1:
        mod_name = model[:last_dot_index]
        class_name = model[last_dot_index + 1:]
        mod = importlib.import_module(mod_name)
        constructor = getattr(mod, class_name)
        return constructor()
    else:
        raise ValueError

def save_results(searcher, results_dir):
    df = pd.DataFrame(searcher.cv_results_)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    file_results = os.path.join(results_dir, "cv_results.csv")
    df.to_csv(file_results, index=False)
    estimator = searcher.best_estimator_
    estim_file = os.path.join(results_dir, "best_estimator.pkl")
    pickle.dump(estimator, open(estim_file, 'wb'))



