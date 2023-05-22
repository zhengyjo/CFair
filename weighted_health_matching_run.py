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
parser.add_argument('-c', '--cov', help='Specify a file for the learned covariance matrix')


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
df_whole = pd.concat([df_black,df_white]).reset_index()

# You can choose any features you want to select
measure_columns = ['rrt', 'gcs', 'sofa_24hours', 'heart_rate', 'sbp_art', 'dbp_art', 'mbp_cuff', 'resp_rate',
                   'temperature', 'spo2', 'glucose', 'gender', 'admission_age', 'charlson_comorbidity_index', 'apsiii',
                   'bmi']



# Transform the candidate list into a dictionary form
res = {}
for j in range(len(candidate_index_lst)):
    if len(candidate_index_lst[j][0]) > 0:
        res[j] = candidate_index_lst[j][0]

# Calculate the initial weighted covariance matrix within the remaining pooled candidates
C_0_std_filter,C_1_std_filter  = filter_with_std_delta_group(df_black_ada,df_white_ada,df_whole,res,measure_columns,coef=1)         
group_zero_delta, group_one_delta = delta_group_construction_from_std_filter(C_0_std_filter,C_1_std_filter)
container_zero_delta, container_one_delta = retrieve_cov_features(group_zero_delta, group_one_delta, df_black, df_white,
                                                                  measure_columns)
candidates, weight = get_covariance_candidate_and_weight(container_zero_delta, container_one_delta)
mab_cov = np.cov(candidates.T, ddof=0, aweights=weight)
mab_cov_inv = np.linalg.inv(mab_cov)

# If we choose to use the learned matrix. Then load the learned matrix
if args.cov:
    cov_dir = args.cov
    mab_cov_inv = torch.load(cov_dir, map_location=torch.device('cpu'))
    mab_cov_inv = cov.numpy()

 
matched_health_distance = matched_health_condition_filtered(df_black,df_white,measure_columns,C_0_std_filter,C_1_std_filter,
                                                        mab_cov_inv)
name = input_file + "_weighted_health_match"

name_1_1 = input_file + "_weighted_health_match_1_1"

with open(name, "wb") as fp:  # Pickling
    pickle.dump(matched_health_distance, fp)

# Perform greedy search for each remaining candidates
one_to_one_counterparts = {}
for index in range(len(C_0_std_filter)):
    res = []
    lst = matched_health_distance[C_0_std_filter[index]]
    if len(lst) > 0:
        lst_np = np.array(lst)
        min_index = np.argmin(lst_np)
        res.append(C_1_std_filter[index][min_index])
        min = np.min(lst_np)
        res.append(min)
    one_to_one_counterparts[C_0_std_filter[index]] = res

with open(name_1_1, "wb") as fp:  # Pickling
    pickle.dump(one_to_one_counterparts, fp)



