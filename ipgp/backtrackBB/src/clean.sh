#!/usr/bin/env bash

echo "Cleaning..."

python3 setup.py clean --all

rm -rf backtrackbb.egg-info
rm -rf backtrackbb/libs/*.so
find | grep __pycache__ | xargs rm -rf

echo "Finished"
