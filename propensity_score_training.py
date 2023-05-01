import pandas as pd
import numpy as np
from datetime import timedelta
from pandas.api.types import CategoricalDtype
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
import pickle
import argparse
import random


# stats test
from scipy.stats import f
import scipy.stats as stats
drive_dir = './'

# The helper function
from calculation import *
import sampling
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression


# create the parser object
parser = argparse.ArgumentParser(description='Example argparse script')

# add arguments to the parser
parser.add_argument('input_file', help='Path to csv file including two ethic groups')


# parse the arguments from the command line
args = parser.parse_args()

# access the argument values
input_file = args.input_file


df = pd.read_csv(input_file)
df = df.drop(['Unnamed: 0'],axis=1)
df = df[(df['ventilation_status'] == 0) |(df['ventilation_status'] == 1)| (df['ventilation_status'] == 5)]

df['ventilation_status'] = df.apply(lambda x: 0 if x['ventilation_status']==0 else (1 if x['ventilation_status']==1 else 2),axis=1)

"""
Normalize the feature values before training the propensity score model

"""

df_selected_mean = df_whole.mean()
df_selected_std = df_whole.std()

original_len = len(df.columns)
for name in df.columns[1:-1]:
    new_name = name + "_nor"
    df[new_name] = \
        df.apply(lambda x: ((x[name] - df_selected_mean[name]) / df_selected_std[name]),
                       axis=1)

df_confounders = df[df.columns[original_len+1:]]
racial = df["race_white_black"]

logistic = LogisticRegression(solver='liblinear')
logistic.fit(df_confounders, racial)

pscore = logistic.predict_proba(df_confounders)[:, 1]
df['propensity_score'] = pscore
df['propensity_logit'] = df['propensity_score'].apply(
    lambda p: np.log(p/(1-p)))

df_white = df_whole[df_whole['race_white_black'] == 0].reset_index()
df_black = df_whole[df_whole['race_white_black'] == 1].reset_index()

df_white.to_csv("white.csv")
df_blak.to_csv("black.csv")

