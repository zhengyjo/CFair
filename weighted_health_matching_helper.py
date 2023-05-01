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


def delta_group_construction(candidate_index_lst):
    group_zero_delta = set()
    group_one_delta = set()

    for i in range(len(candidate_index_lst)):
        candidate = candidate_index_lst[i][0]
        if len(candidate) > 0:
            group_zero_delta.add(i)
            for ind in candidate:
                group_one_delta.add(ind)
    return group_zero_delta, group_one_delta


def retrieve_cov_features(group_zero_delta, group_one_delta, df_black, df_white, feature_columns):
    container_zero_delta = []
    container_one_delta = []
    for minor in group_zero_delta:
        minor_candidate = df_black.iloc[minor][feature_columns].to_numpy()
        container_zero_delta.append(minor_candidate)
    for major in group_one_delta:
        major_candidate = df_white.iloc[major][feature_columns].to_numpy()
        container_one_delta.append(major_candidate)
    return np.array(container_zero_delta), np.array(container_one_delta)


def get_covariance_candidate_and_weight(container_zero_delta, container_one_delta):
    w_zero = 1 / 2 / len(container_zero_delta)
    w_one = 1 / 2 / len(container_one_delta)
    W_zero = [w_zero] * (container_zero_delta.shape[0])
    W_one = [w_one] * (container_one_delta.shape[0])
    weight = np.hstack([W_zero, W_one])
    candidates = np.vstack([container_zero_delta, container_one_delta])
    return candidates, weight


def all_matched_health_condition(df_black, df_white, measure_columns, candidate_index_lst, mab_cov_inv):
    matched_health_distance = []
    for index, row in df_black.iterrows():
        black_score = row[measure_columns].to_numpy()
        candidates = candidate_index_lst[index][0]
        candidate_distance_lst = []
        if len(candidates) > 0:
            for white_index in candidates:
                white_id = df_white.iloc[white_index]['subject_id']
                white_score = df_white.iloc[white_index][measure_columns].to_numpy()
                temp_dist = distance.mahalanobis(black_score, white_score, mab_cov_inv)
                candidate_distance_lst.append(temp_dist)
            matched_health_distance.append(candidate_distance_lst)
    return matched_health_distance
