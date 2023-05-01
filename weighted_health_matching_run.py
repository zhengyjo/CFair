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
from weighted_health_matching_helper import *
from scipy.spatial import distance

# create the parser object
parser = argparse.ArgumentParser(description='Example argparse script')

# add arguments to the parser
parser.add_argument('input_file', help='Path to input file')
parser.add_argument('minority_file', help='Path to minority csv file')
parser.add_argument('majority_file', help='Path to majority csv file')


# parse the arguments from the command line
args = parser.parse_args()

# access the argument values
input_file = args.input_file
minority_file = args.minority_file
majority_file = args.majority_file

with open(input_file, "rb") as fp:  # Unpickling
    candidate_index_lst = pickle.load(fp)

df_black = pd.read_csv(minority_file)
df_white = pd.read_csv(majority_file)

measure_columns = ['rrt', 'gcs', 'sofa_24hours', 'heart_rate', 'sbp_art', 'dbp_art', 'mbp_cuff', 'resp_rate',
                   'temperature', 'spo2', 'glucose', 'gender', 'admission_age', 'charlson_comorbidity_index', 'apsiii',
                   'bmi']

group_zero_delta, group_one_delta = delta_group_construction(candidate_index_lst)
container_zero_delta, container_one_delta = retrieve_cov_features(group_zero_delta, group_one_delta, df_black, df_white,
                                                                  measure_columns)
candidates, weight = get_covariance_candidate_and_weight(container_zero_delta, container_one_delta)
mab_cov = np.cov(candidates.T, ddof=0, aweights=weight)
mab_cov_inv = np.linalg.inv(mab_cov)
matched_health_distance = all_matched_health_condition(df_black, df_white, measure_columns,
                                                       candidate_index_lst, mab_cov_inv)
name = input_file + "_weighted_health_match"

name_1_1 = input_file + "_weighted_health_match_1_1"

with open(name, "wb") as fp:  # Pickling
    pickle.dump(matched_health_distance, fp)

one_to_one_counterparts = []
for lst in matched_health_distance:
    res = []
    if len(lst) > 0:
        lst_np = np.array(lst)
        min_index = np.argmin(lst_np)
        res.append(min_index)
        min = np.min(lst_np)
        res.append(min)
    one_to_one_counterparts.append(res)

with open(name_1_1, "wb") as fp:  # Pickling
    pickle.dump(one_to_one_counterparts, fp)



