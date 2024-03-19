#!/usr/bin/env python
import os
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from jax.experimental import optimizers
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


DATA_LOGIC50 = "https://raw.githubusercontent.com/saezlab/Macau_project_1/master/DATA/IC50"
DATA_DRUG_FEATURES = "https://raw.githubusercontent.com/saezlab/Macau_project_1/master/DATA/target"
DATA_CELL_FEATURES = "https://raw.githubusercontent.com/saezlab/Macau_project_1/master/DATA/progeny11"

def load_ic50(file):
    df_logIC50 = pd.read_csv(file)
    df_logIC50 = df_logIC50.rename(columns={df_logIC50.columns[0]: 'drug'}).set_index('drug')
    df_logIC50.columns = df_logIC50.columns.astype(int)
    df_logIC50.columns.name = 'cell'
    df_logIC50 = df_logIC50.dropna(how='all', axis=0).dropna(how='all', axis=1)
    df_logIC50 = df_logIC50.groupby(level=0).mean() # merge duplicates
    return df_logIC50

def load_drug_features(file):
    df_drug_features = pd.read_csv(file)
    df_drug_features = df_drug_features.rename(columns={df_drug_features.columns[0]: 'drug'}).set_index('drug')
    df_drug_features = df_drug_features.groupby(level=0).first() # merge dups
    return df_drug_features

def load_cell_features(file):
    df_cell_features = pd.read_csv(file)
    df_cell_features = df_cell_features.rename(columns={df_cell_features.columns[0]: 'cell'}).set_index('cell')
    df_cell_features = df_cell_features.add_prefix("PROGENY_").reset_index().rename(columns={'index':'cell'}).set_index('cell')
    return df_cell_features

def split(df_response, df_row_feats=None, df_col_feats=None, frac_rows=0.1, frac_cols=0.1):
    test_rows = df_response.sample(frac=frac_rows, replace=False).index
    test_cols = df_response.T.sample(frac=frac_cols, replace=False).index
    train_rows = df_response.index.difference(test_rows)
    train_cols = df_response.columns.difference(test_cols)
    df_response_test = df_response.loc[test_rows, test_cols]
    df_response_train = df_response.loc[train_rows, train_cols]
    if df_row_feats is not None:
        df_row_feats_train = df_row_feats.loc[train_rows, :]
        df_row_feats_test = df_row_feats.loc[test_rows, :]
    else:
        df_row_feats_train = df_row_feats_test = None
        df_response_test = df_response.loc[:, test_cols]

    if df_col_feats is not None:
        df_col_feats_train = df_col_feats.loc[train_cols, :]
        df_col_feats_test = df_col_feats.loc[test_cols, :]
    else:
        df_col_feats_train = df_col_feats_test = None
        df_response_test = df_response.loc[test_rows, :]
    return ([df_response_train, df_row_feats_train, df_col_feats_train], 
            [df_response_test, df_row_feats_test, df_col_feats_test])

def kfold_validation(df_ic50, drug_features=None, cell_features=None, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True)
    if drug_features is not None:
        pass
    if cell_features is not None:
        pass
    # Go through all:
    pass


def initialize_weights(data, row_features=None, col_features=None, k=10):
    if row_features is not None:
        LD = np.random.normal(size=(k, row_features.shape[1]))
    else:
        LD = np.random.normal(size=(k, data.shape[0]))
    if col_features is not None:
        LC = np.random.normal(size=(k, col_features.shape[1]))
    else:
        LC = np.random.normal(size=(k, data.shape[1]))
    ld_bias = jnp.zeros((k, 1))
    lc_bias = jnp.zeros((k, 1))
    mu = 0.0
    return [LD, LC, ld_bias, lc_bias, mu]

@jax.jit
def mf(params):
    LD, LC, ld_bias, lc_bias, mu = params
    Dt = jnp.transpose(jnp.add(LD, ld_bias))
    C = jnp.add(LC, lc_bias)
    return jnp.dot(Dt, C) + mu

@jax.jit
def mf_with_row_features(params, row_features):
    LD, LC, ld_bias, lc_bias, mu = params
    D = jnp.add(jnp.dot(LD, jnp.transpose(row_features)), ld_bias)
    Dt = jnp.transpose(D)
    C = jnp.add(LC, lc_bias)
    return jnp.dot(Dt, C) + mu

@jax.jit
def mf_with_col_features(params, col_features):
    LD, LC, ld_bias, lc_bias, mu = params
    Dt = jnp.transpose(jnp.add(LD, ld_bias))
    C = jnp.add(jnp.dot(LC, jnp.transpose(col_features)), lc_bias)
    return jnp.dot(Dt, C) + mu

@jax.jit
def mf_with_features(params, row_features, col_features):
    LD, LC, ld_bias, lc_bias, mu = params
    Dt = jnp.transpose(jnp.add(jnp.dot(LD, jnp.transpose(row_features)), ld_bias)) 
    C = jnp.add(jnp.dot(LC, jnp.transpose(col_features)), lc_bias)
    return jnp.dot(Dt, C) + mu

# Implementation of MSE loss ignoring NaN values
@jax.jit
def loss_mse(X, X_hat):
    # Count the number of valid values in the matrix
    is_nan = jnp.isnan(X)
    n = jnp.sum(~is_nan)
    # Replace NaNs with 0s. It does not affect the loss
    # as we're going to compute the average ignoring 0s
    Xf = jnp.nan_to_num(X, nan=0.)
    # Put 0s on NaN positions
    X_hat_f = jnp.where(is_nan, 0., X_hat)
    # Sum of squared residuals
    sq = jnp.power(Xf - X_hat_f, 2)
    # Average using non missing entries
    return jnp.sum(sq) / n

@jax.jit
def predict(params, row_features=None, col_features=None):
    if row_features == None and col_features == None:
        X_hat = mf(params)
    elif row_features != None and col_features == None:
        X_hat = mf_with_row_features(params, row_features)
    elif col_features != None and row_features == None:
        X_hat = mf_with_col_features(params, col_features)
    else:
        X_hat = mf_with_features(params, row_features, col_features)
    return X_hat

@jax.jit
def loss_mf(params, X, row_features=None, col_features=None, reg=0.0):
    X_hat = predict(params, row_features, col_features)
    # Add regularization for latent matrices
    l2_ld = jnp.sum(jnp.power(params[0], 2))
    l2_lc = jnp.sum(jnp.power(params[1], 2))
    return loss_mse(X, X_hat) + reg*(l2_ld + l2_lc)

def optimize(X, params, opt=optimizers.adam(0.1), loss_fn=loss_mf, 
             loss_options=dict(), epochs=1000):
    opt_state = opt.init_fn(params)
    steps = tqdm(range(epochs))
    for step in steps:
        value, grads = jax.value_and_grad(loss_fn)(opt.params_fn(opt_state), X, **loss_options)
        opt_state = opt.update_fn(step, grads, opt_state)
        steps.set_postfix({'loss': "{:.4f}".format(value)})
    return opt.params_fn(opt_state)

def r2(y, y_hat):
    yf = np.array(y).flatten()
    yf_hat = np.array(y_hat).flatten()
    isnan = np.isnan(yf)
    yf = yf[~isnan]
    yf_hat = yf_hat[~isnan]
    r2 = np.nan
    ss_res = np.nansum((yf - yf_hat)**2)
    ss_total = np.nansum((yf - np.nanmean(yf))**2)
    if ss_total > 0:
        r2 = 1 - ss_res/ss_total
    return r2


          

if __name__ == "__main__":
    print("Using JAX version", jax.__version__)
    
    parser = argparse.ArgumentParser(description="ML UC2 model")
    parser.add_argument('input_file', type=str, help='IC50 csv response data for training or npz file with the model for inference')
    parser.add_argument('output_file', type=str, help='File to store predictions in inference mode or npz model if in training mode')
    parser.add_argument('--drug_features', type=str, default=DATA_DRUG_FEATURES, help="File with drug features")
    parser.add_argument('--cell_features', type=str, default=DATA_CELL_FEATURES, help="File with cell features")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--adam_lr', type=float, default=0.1, help='Learning rate for ADAM optimizer')
    parser.add_argument('--reg', type=float, default=1e-3, help='Regularization penalty for sparisty')
    parser.add_argument('--test_drugs', type=float, default=0.0, help='Proportion of drugs removed from training and used for test')
    parser.add_argument('--test_cells', type=float, default=0.0, help='Proportion of cell lines removed from training and used for test')
    parser.add_argument('--latent_size', type=int, default=10, help='Size of the latent vector')
    parser.add_argument('--folds', type=int, default=0, help='Number of folds for k-fold validation strategy (ignores test_drugs and test_cells)')
    args = parser.parse_args()
    

    df_logIC50 = None
    if args.input_file.endswith('.x'):
        print(f'Using example log(IC50) from {DATA_LOGIC50}')
        df_logIC50 = load_ic50(DATA_LOGIC50)
    else:
        if not args.input_file.endswith('.npz'):
            df_logIC50 = load_ic50(args.input_file)

    if args.drug_features.endswith('.x'):
        print(f'Using example drug features from {DATA_DRUG_FEATURES}')
        df_drug_features = load_drug_features(DATA_DRUG_FEATURES)
    elif args.drug_features is None or args.drug_features.endswith('.none'):
        df_drug_features = None
        if args.test_drugs > 0.0:
            print("No features provided for drugs, cannot test drug predictions, setting test_drugs to 0.0")
            args.test_drugs = 0.0
    else:
        print(f'Using drug features from {args.drug_features}')
        df_drug_features = load_drug_features(args.drug_features)

    if args.cell_features.endswith('.x'):
        print(f'Using example cell features from {DATA_CELL_FEATURES}')
        df_cell_features = load_drug_features(DATA_CELL_FEATURES)
    elif args.cell_features is None or args.cell_features.endswith('.none'):
        df_cell_features = None
        if args.test_cells > 0.0:
            print("No features provided for cells, cannot test cell predictions, setting test_cells to 0.0")
            args.test_cells = 0.0
    else:
        print(f'Using cell features from {args.cell_features}')
        df_cell_features = load_cell_features(args.cell_features)

    if df_drug_features is not None and df_logIC50 is not None:
        common_drugs = df_drug_features.index.intersection(df_logIC50.index)
        df_drug_features = df_drug_features.loc[common_drugs, :] # align with drug IDs
        df_logIC50 = df_logIC50.loc[common_drugs, :]
        print("Drug features", df_drug_features.shape)
    if df_cell_features is not None and df_logIC50 is not None:
        common_cells = df_cell_features.index.intersection(df_logIC50.columns.astype(int))
        df_cell_features = df_cell_features.loc[common_cells, :]
        df_logIC50 = df_logIC50.loc[:, common_cells]
        print("Cell features", df_cell_features.shape)
    if df_logIC50 is not None:    
        print(f'Response size after alignment: {df_logIC50.shape}')


    if args.input_file.endswith('.npz'):
        print(f"Loading model from file {args.input_file}...")
        p = np.load(args.input_file, allow_pickle=True)
        params = [p['LD'], p['LC'], p['ld_bias'], p['lc_bias'], p['mu']]
        # predict
        row_features = df_drug_features.to_numpy() if df_drug_features is not None else None
        col_features = df_cell_features.to_numpy() if df_cell_features is not None else None
        idx = df_drug_features.index if row_features is not None else df_logIC50.index
        cols = df_cell_features.index if col_features is not None else df_logIC50.columns
        X_hat = predict(params, row_features=row_features, col_features=col_features)
        df_pred = pd.DataFrame(X_hat, index=idx, columns=cols)
        print(df_pred)
        print(f"Saving to {args.output_file}...")
        df_pred.to_csv(args.output_file)
    else:
        print(f"Using a latent vector of size {args.latent_size}.")
        print(f"Using ADAM with lr={args.adam_lr}, epochs={args.epochs}, l2 regularization={args.reg}")
        print(f"Response data of size {df_logIC50.shape} (drugs x cells)")
        train, test = split(df_logIC50, df_row_feats=df_drug_features, df_col_feats=df_cell_features, frac_rows=args.test_drugs, frac_cols=args.test_cells)
        [df_response_train, df_drug_train, df_cell_train] = train
        [df_response_test, df_drug_test, df_cell_test] = test
        print(f'Keeping {args.test_drugs} rows for test, and {args.test_cells} cols for test')
        print(f'New training data size: {df_response_train.shape}')
        row_features = df_drug_train.to_numpy() if df_drug_train is not None else None
        col_features = df_cell_train.to_numpy() if df_cell_train is not None else None
        params = initialize_weights(df_response_train, row_features=row_features, col_features=col_features, k=args.latent_size)
        opt = optimizers.adam(args.adam_lr)
        params = optimize(df_response_train.to_numpy(), params, epochs=args.epochs, opt=opt, 
                        loss_options={'row_features': row_features, 'col_features': col_features, 'reg': args.reg})
        LD, LC, ld_bias, lc_bias, mu = params
        if df_response_test is not None:
            print(f'Test response shape: {df_response_test.shape}')
            row_features = df_drug_test.to_numpy() if df_drug_test is not None else None
            col_features = df_cell_test.to_numpy() if df_cell_test is not None else None
            X_hat = predict(params, row_features=row_features, col_features=col_features)
            # Calculate baseline
            if row_features is not None and col_features is None:
                # Unknown drugs, known cells
                mean_cols = df_response_train.mean(axis=0).fillna(df_response_train.mean().mean())
                X_baseline = pd.concat([mean_cols]*X_hat.shape[0], axis=1).to_numpy().T
            elif row_features is None and col_features is not None:
                # Unknown cells, known drugs
                mean_rows = df_response_train.mean(axis=1)
                X_baseline = pd.concat([mean_rows]*X_hat.shape[1], axis=1).to_numpy()
            elif row_features is not None and col_features is not None:
                # For unknown rows/features for which we dont IC50 values, baseline prediction is just
                # the average of the log(IC50) in the training set.
                X_baseline = np.full_like(X_hat, df_response_train.mean().mean())
            X = df_response_test.to_numpy()
            print(f'Test prediction shape: {X_hat.shape}')
            e = loss_mse(X, X_hat)
            eb = loss_mse(X, X_baseline)
            rsq = r2(X, X_hat)
            print(f"MSE_test: {e:.4f}, R2_test: {rsq:.4f}, MSE_baseline: {eb:.4f}")
        print(f"Exporting model to {args.output_file}...")
        np.savez_compressed(args.output_file, LD=LD, LC=LC, ld_bias=ld_bias, lc_bias=lc_bias, mu=mu)
    print("Done.")
        
