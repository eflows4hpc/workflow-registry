#!/usr/bin/env python
import argparse
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature merger for CARNIVAL")
    parser.add_argument('folder', type=str, help="Path containing the folders with the samples. Name of the folders are used for the name of the samples")
    parser.add_argument('output', type=str, help="Output file with the features, where rows are samples and columns features")
    parser.add_argument('--feature_file', type=str, default=None, help="File containing a list of features. If provided, only those features are retrieved from solutions.")
    parser.add_argument('--merge_csv_file', type=str, default=None, help="If provided, join the merged features into the given file.")
    parser.add_argument('--merge_csv_index', type=str, default="sample", help="If provided, join the merged features into the given file.")
    parser.add_argument('--merge_csv_prefix', type=str, default="F_", help="Prefix for the merged features")
    args = parser.parse_args()

    samples = [s for s in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, s))]
    print(f"{len(samples)} found")
    feats = []
    if args.feature_file is not None and os.path.isfile(args.feature_file):
        with open(args.feature_file) as f:
            feats = [line.rstrip() for line in f if len(line.strip()) > 0]
        print(f"Loaded {len(feats)} features from {args.feature_file}")

    dataframes = []
    missing = 0
    for s in samples:
        print(f"Processing {s}...")
        fpath = os.path.join(args.folder, s, 'carnival.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df.set_index(df.columns[0], inplace = True)
            df.rename(columns={df.columns[0]: s}, inplace = True)
            dataframes.append(df)
        else:
            print("Missing carnival.csv file")
            missing += 1
    df_result = pd.concat(dataframes, axis=1, verify_integrity=True).T
    df_result.index = df_result.index.astype(str)
    print(f"Missing samples: {missing}")
    print(f"Features merged, shape: {df_result.shape}")
    if feats:
        df_result = df_result.loc[:, feats]
        print(f"Selected a subset of features, current shape: {df_result.shape}")
    if args.merge_csv_file:
        df_ext = pd.read_csv(args.merge_csv_file, dtype={args.merge_csv_index: 'object'})
        df_ext.set_index(args.merge_csv_index, inplace=True)
        df_ext = df_ext.add_prefix(args.merge_csv_prefix)
        print(df_ext)
        common = df_ext.index.intersection(samples)
        df_ext = df_ext.loc[common, :]
        print(f"Loaded csv from {args.merge_csv_file}, shape: {df_ext.shape}")
        df_result = df_result.join(df_ext)
        print(f"Features merged with external csv, new shape: {df_result.shape}")
    print(f"Exporting to {args.output}...")
    df_result.to_csv(args.output)
    print("Done.")
