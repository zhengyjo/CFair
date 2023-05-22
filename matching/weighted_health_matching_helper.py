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
    """
    Extract the remaining majority and minority patients from the candidate list
    Parameters:
    -------
    candidate_index_lst: List of lists. The i,j entry represents the ith minority patients has the jth majority match after the previous stage
    of candidate filtering.

    Return :
    -------
    group_zero_delta : set. It contatins the indexes of remaining minority patients from the minority dataframe
    group_one_delta : set. It contains the indexes of remaining majority patients from the majority dataframe
    """
    group_zero_delta = set()
    group_one_delta = set()

    for i in range(len(candidate_index_lst)):
        candidate = candidate_index_lst[i][0]
        if len(candidate) > 0:
            group_zero_delta.add(i)
            for ind in candidate:
                group_one_delta.add(ind)
    return group_zero_delta, group_one_delta

def filter_with_std_delta_group(df_black,df_white,df_whole,candidate_lst_dict,measure_columns,coef=1):
    """
    Extract the remaining majority and minority patients from the candidate list after filtering
    Parameters:
    -------
    df_black : dataframe. The dataframe that stores the information of minority patients.
    df_white : dataframe. The dataframe that stores the information of majority patients.
    df_whole : dataframe. The dataframe that stores the information of both majority and minority patients.
    feature_columns : list. The features user want to use for covariance calcualtion. 
    Return :
    -------
    C_0_filter_std : list. It contatins the indexes of remaining minority patients from the minority dataframe
    C_1_filter_std : list of lists. The i,j entry represents the ith minority patients has the jth majority match after the previous stage
    of candidate filtering.
    """
    std = df_whole.std()[measure_columns]
    C_0_filter_std = []
    C_1_filter_std = []
    for key in candidate_lst_dict.keys():
        black_features = df_black.iloc[key][measure_columns] 
        left_white_match = []
        for white_index in candidate_lst_dict[key]:
            white_features = df_white.iloc[white_index][measure_columns]
            flag = 1
            for ele in measure_columns:
                white_ele = float(white_features[ele])
                black_ele = float(black_features[ele])
                std_ele = float(std[ele])
                if (abs(black_ele - white_ele)) > coef * std_ele:
                    flag = 0
                    break
            if flag == 1:
                left_white_match.append(white_index)
                print("finally!")
        if (len(left_white_match) > 0):
            C_0_filter_std.append(key)
            C_1_filter_std.append(left_white_match)
        print(key)
    return C_0_filter_std,C_1_filter_std

def delta_group_construction_from_std_filter(C_0_std_filter, C_1_std_filter):
    """
    Extract the remaining majority and minority patients from the candidate list
    Parameters:
    -------
    C_0_std_filter: list. It contatins the indexes of remaining minority patients from the minority dataframe.
    C_1_std_filter: list of lists. The i,j entry represents the ith minority patients has the jth majority match after the previous stage
    of candidate filtering.

    Return :
    -------
    group_zero_delta : set. It contatins the indexes of remaining minority patients from the minority dataframe
    group_one_delta : set. It contains the indexes of remaining majority patients from the majority dataframe
    """
    group_zero_delta = set()
    group_one_delta = set()

    for i in range(len(C_0_std_filter)):
        candidates = C_1_std_filter[i]
        if len(candidates) > 0:
            group_zero_delta.add(C_0_std_filter[i])
            for ind in candidates:
                group_one_delta.add(ind)
    return group_zero_delta, group_one_delta

def retrieve_cov_features(group_zero_delta, group_one_delta, df_black, df_white, feature_columns):
     """
    Extract the essential information for calculating the weighted covariance matrix.
    Parameters:
    -------
    group_zero_delta : set. It contatins the indexes of remaining minority patients from the minority dataframe
    group_one_delta : set. It contains the indexes of remaining majority patients from the majority dataframe
    df_black : dataframe. The dataframe that stores the information of minority patients.
    df_white : dataframe. The dataframe that stores the information of majority patients.
    feature_columns : list. The features user want to use for covariance calcualtion. 

    Return :
    -------
    container_zero_delta : numpy array. It contatins the feature of the remaining miority patients
    container_one_delta : numpy array. It contains the feature of the remaining majority patients.
    """
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
     """
    Assign weights to the remaining patients
    Parameters:
    -------
    container_zero_delta : numpy array. It contatins the feature of the remaining miority patients
    container_one_delta : numpy array. It contains the feature of the remaining majority patients.

    Return :
    -------
    candidates : numpy array. It contains the information of both minority and majority patients.
    weight : numpy array. It contains the calculated weight for each patients.
    """
    w_zero = 1 / 2 / len(container_zero_delta)
    w_one = 1 / 2 / len(container_one_delta)
    W_zero = [w_zero] * (container_zero_delta.shape[0])
    W_one = [w_one] * (container_one_delta.shape[0])
    weight = np.hstack([W_zero, W_one])
    candidates = np.vstack([container_zero_delta, container_one_delta])
    return candidates, weight


   
def matched_health_condition_filtered(df_minor, df_major, measure_columns, C_0_vent_lst,C_1_lst, mab_cov_inv):
     """
    Perform mahalanobis distance calculation between each remaining minority patient and his/her corresponding match candidates.
    Parameters:
    -------
    df_black : dataframe. The dataframe that stores the information of minority patients.
    df_white : dataframe. The dataframe that stores the information of majority patients.
    measure_columns: list. The feature user want to use in greedy search.
    C_0_vent_lst: list. The list of indexes of remaining minority patient. 
    C_1_lst: The list of indexes of remaining majority patient.
    mab_cov_inv: numpy array. The matrix use want to use for mahalanobis distance calculation.

    Return :
    -------
    matched_health_distance : dict. It contains the distance information between each remaining minority patientand his/her 
    majority matches.
    """
    matched_health_distance = {}
    for index in range(len(C_0_vent_lst)):
        black_feature = df_minor.iloc[C_0_vent_lst[index]]
        black_score = black_feature[measure_columns].to_numpy()
        candidates = C_1_lst[index]
        candidate_distance_lst = []
        if len(candidates) > 0:
            for white_index in candidates:
                white_score = df_major.iloc[white_index][measure_columns].to_numpy()
                temp_dist = distance.mahalanobis(black_score, white_score, mab_cov_inv)
                candidate_distance_lst.append(temp_dist)
            matched_health_distance[C_0_vent_lst[index]] = candidate_distance_lst
    return matched_health_distance
 
 
