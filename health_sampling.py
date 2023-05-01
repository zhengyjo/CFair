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


def retrieve_cov_elements(candidate_lst):
    res = {}
    for i in range(len(candidate_lst)):
        candidate = candidate_lst[i][0]
        if (len(candidate) > 0):
            # Randomly sample one elements
            ind = random.randint(0, len(candidate)-1)
            res[i] = candidate[ind]
    return res

def retrieve_cov_features(matched_dict,df_black,df_white,feature_columns):
    container = []
    for key in matched_dict.keys():
        minor_candidate = df_black.iloc[key][feature_columns].to_numpy()
        container.append(minor_candidate)
        major_candidate = df_white.iloc[matched_dict[key]][feature_columns].to_numpy()
        container.append(major_candidate)
    return np.array(container)

def all_matched_health_conditoin(df_black,df_white,measure_columns,candidate_index_lst,mab_cov_inv):
    matched_health_distance = []
    for index,row in df_black.iterrows():
        black_score = row[measure_columns].to_numpy()
        candidates = candidate_index_lst[index][0]
        candidate_distance_lst = []
        if len(candidates) > 0:
            for white_index in candidates:
                white_id = df_white.iloc[white_index]['subject_id']
                white_score = df_white.iloc[white_index][measure_columns].to_numpy()
                temp_dist = distance.mahalanobis(black_score,white_score,mab_cov_inv)
                candidate_distance_lst.append(temp_dist)
            matched_health_distance.append(candidate_distance_lst)
    return matched_health_distance


# create the parser object
parser = argparse.ArgumentParser(description='Example argparse script')

# add arguments to the parser
parser.add_argument('input_file', help='Path to input file')

df_black = pd.read_csv("black.csv")
df_white = pd.read_csv("white.csv")
measure_columns = ['rrt', 'gcs','sofa_24hours','heart_rate','sbp_art','dbp_art','mbp_cuff','resp_rate','temperature','spo2','glucose','gender','admission_age','charlson_comorbidity_index','apsiii','bmi']

# parse the arguments from the command line
args = parser.parse_args()

# access the argument values
input_file = args.input_file


with open(input_file, "rb") as fp:   # Unpickling
    candidate_index_lst = pickle.load(fp)

for i in range(100):
    print(i)
    mab_idx = retrieve_cov_elements(candidate_index_lst)
    mab_candidates = retrieve_cov_features(mab_idx, df_black, df_white, measure_columns)
    mab_cov = np.cov(mab_candidates.T)
    mab_cov_inv = np.linalg.inv(mab_cov)
    matched_health_distance = all_matched_health_conditoin(df_black, df_white, measure_columns,
                                                               candidate_index_lst, mab_cov_inv)
    name = input_file + "_health_match_"+str(i)
    with open(name, "wb") as fp:  # Pickling
        pickle.dump(matched_health_distance, fp)
