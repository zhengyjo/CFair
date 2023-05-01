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


def filter_with_delta_threshold(distance_psm_all, all_pair_distance_lst, delta):
    """
    First stage of counter-part filtering with given delta
    Parameters:
    -------
    distance_psm_all: the data frame of minority group
    delta: The p-value threshold for null distribution of all the pair-wise difference among pairs with one candidate
    from major group and the other one from minority group

    Return :
    -------
    list.  The indices of qualified neighbor for each minority candidate.
    """
    threshold = np.quantile(all_pair_distance_lst, delta)
    candidate_index_lst = []
    for i in range(len(distance_psm_all)):
        temp_lst = np.where(np.array(distance_psm_all[i]) < threshold)
        candidate_index_lst.append(temp_lst)
    return candidate_index_lst


# create the parser object
parser = argparse.ArgumentParser(description='Example argparse script')
parser.add_argument('input_file', help='The file including all the pair-wise distance')
parser.add_argument('input_file_2', help='The file including respective the pair-wise distance')
parser.add_argument('p_value', help='The chosen p-value threshold')
parser.add_argument('name', help='Save qualified delta-group index')

all_pair_distance_lst = args.input_file
all_pair_distance = args.input_file_2
p_value = float(args.delta)
name = args.name

candidate_index_lst = filter_with_delta_threshold(all_pair_distance, all_pair_distance_lst, p_value)

with open(name, "wb") as fp:  # Pickling
    pickle.dump(candidate_index_lst, fp)
